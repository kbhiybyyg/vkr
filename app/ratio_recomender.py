from __future__ import annotations
import os, json, pickle, math
from typing import List, Optional, Union, Dict, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# =========================================================
# Продвинутая версия под тем же именем
# =========================================================
class RatioRecommenderTF:
    def __init__(
        self,
        n_estimators: int = 3,          # число моделей в ансамбле
        hidden_main: int = 160,         # ширина основного блока
        proj_units: int = 80,           # сжатие перед выходом
        n_resblocks: int = 3,           # число residual-блоков
        dropout: float = 0.10,
        l2: float = 1e-4,
        base_lr: float = 8e-4,          # базовый lr (дальше управляет шедулер)
        epochs: int = 800,              # большие эпохи + ранняя остановка
        batch_size: int = 64,
        val_size: float = 0.25,
        random_state: int = 42,
        verbose: int = 1,
        test_size: float = 0.30,        # для метрик holdout
    ):
        self.n_estimators = int(max(1, n_estimators))
        self.hidden_main = int(hidden_main)
        self.proj_units = int(proj_units)
        self.n_resblocks = int(n_resblocks)
        self.dropout = float(dropout)
        self.l2 = float(l2)
        self.base_lr = float(base_lr)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.val_size = float(val_size)
        self.random_state = int(random_state)
        self.verbose = int(verbose)
        self.test_size = float(test_size)

        # заполнится после fit/load
        self.models: List[keras.Model] = []
        self.scaler: Optional[RobustScaler] = None
        self.feature_order: Optional[List[str]] = None
        self.target_col: Optional[str] = None
        self.y_mean_: Optional[float] = None
        self.y_std_: Optional[float] = None
        self.history_: List[dict] = []
        self.metrics_: Optional[dict] = None

        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)

    # ======== публичные ========
    def fit(self, df: pd.DataFrame, target: str, features: Optional[List[str]] = None) -> "RatioRecommenderTF":
        self.target_col = target
        if not features:
            features = (
                df.drop(columns=[target], errors="ignore")
                  .select_dtypes(include=[np.number])
                  .columns.tolist()
            )
        if target in features:
            features.remove(target)
        self.feature_order = features

        X = df[features].astype(float).values
        y = pd.to_numeric(df[target], errors="coerce").astype(float).values
        mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        X, y = X[mask], y[mask]

        X_tr, X_te, y_tr, y_te = train_split(
            X, y, test_size=self.test_size, seed=self.random_state
        )

        # скейлер на трене
        self.scaler = RobustScaler()
        X_tr = self.scaler.fit_transform(X_tr)
        X_te = self.scaler.transform(X_te)

        # нормировка таргета (z-score)
        self.y_mean_, self.y_std_ = float(np.mean(y_tr)), float(np.std(y_tr) + 1e-8)
        y_tr_z = (y_tr - self.y_mean_) / self.y_std_
        y_te_z = (y_te - self.y_mean_) / self.y_std_

        self.models = []
        self.history_ = []

        # ансамбль: бутстрэп + разные сиды
        n = X_tr.shape[0]
        rng = np.random.default_rng(self.random_state)

        for m in range(self.n_estimators):
            idx = rng.choice(n, size=n, replace=True)
            X_b, y_b = X_tr[idx], y_tr_z[idx]

            model = self._build_model(X_tr.shape[1], seed=self.random_state + m)

            # шедулер: косинус с рестартами — не «залипает», учится стабильно
            steps_per_epoch = max(1, math.ceil(len(X_b) / self.batch_size))
            schedule = keras.optimizers.schedules.CosineDecayRestarts(
                initial_learning_rate=self.base_lr,
                first_decay_steps=max(200, 10 * steps_per_epoch),
                t_mul=1.75,
                m_mul=0.95,
                alpha=0.05,
            )
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=schedule, amsgrad=True),
                loss=keras.losses.Huber(delta=1.0),
                metrics=["mae"],
            )

            callbacks = [
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=120,               # чтобы не «рубило» рано
                    restore_best_weights=True,
                    verbose=self.verbose,
                    min_delta=1e-4
                )
            ]

            hist = model.fit(
                X_b, y_b,
                validation_split=self.val_size,
                epochs=self.epochs,
                batch_size=self.batch_size,
                shuffle=True,
                callbacks=callbacks,
                verbose=self.verbose
            )
            self.models.append(model)
            self.history_.append({k: [float(v) for v in vals] for k, vals in hist.history.items()})

        # метрики на holdout в ИСХОДНЫХ единицах
        y_pred_z = np.mean([m.predict(X_te, verbose=0).reshape(-1) for m in self.models], axis=0)
        y_pred = y_pred_z * self.y_std_ + self.y_mean_
        rmse = float(np.sqrt(np.mean((y_te - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_te - y_pred)))

        self.metrics_ = {
            "rmse": rmse,
            "mae": mae,
            "n_train": int(X_tr.shape[0]),
            "n_test": int(X_te.shape[0]),
            "input_dim": int(X_tr.shape[1]),
            "n_estimators": int(self.n_estimators),
        }
        return self

    def predict(self, X: Union[pd.DataFrame, np.ndarray, List[Dict]]) -> np.ndarray:
        self._check_ready()
        Xp = self._prepare_X(X)
        z = np.mean([m.predict(Xp, verbose=0).reshape(-1) for m in self.models], axis=0)
        return z * self.y_std_ + self.y_mean_

    def predict_one(self, sample: Dict) -> float:
        return float(self.predict([sample])[0])

    def save(self, folder: str) -> None:
        self._check_ready()
        os.makedirs(folder, exist_ok=True)
        # модели
        for i, m in enumerate(self.models):
            m.save(os.path.join(folder, f"keras_model_{i}.keras"))
        # скейлер
        with open(os.path.join(folder, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)
        # мета
        meta = {
            "feature_order": self.feature_order,
            "target_col": self.target_col,
            "y_norm": {"mean": self.y_mean_, "std": self.y_std_},
            "hyperparams": {
                "n_estimators": self.n_estimators,
                "hidden_main": self.hidden_main,
                "proj_units": self.proj_units,
                "n_resblocks": self.n_resblocks,
                "dropout": self.dropout,
                "l2": self.l2,
                "base_lr": self.base_lr,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "val_size": self.val_size,
                "random_state": self.random_state,
                "verbose": self.verbose,
                "test_size": self.test_size,
            },
            "metrics": self.metrics_,
            "history": self.history_,
        }
        with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, folder: str) -> "RatioRecommenderTF":
        with open(os.path.join(folder, "meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        obj = cls(**meta.get("hyperparams", {}))
        obj.feature_order = meta.get("feature_order")
        obj.target_col = meta.get("target_col")
        yn = meta.get("y_norm") or {}
        obj.y_mean_, obj.y_std_ = yn.get("mean"), yn.get("std")
        obj.metrics_ = meta.get("metrics")
        obj.history_ = meta.get("history") or []
        # скейлер
        with open(os.path.join(folder, "scaler.pkl"), "rb") as f:
            obj.scaler = pickle.load(f)
        # модели
        obj.models = []
        i = 0
        while True:
            path = os.path.join(folder, f"keras_model_{i}.keras")
            if not os.path.exists(path):
                break
            obj.models.append(keras.models.load_model(path))
            i += 1
        if not obj.models:
            # fallback: одна модель под старым именем
            single = os.path.join(folder, "keras_model.keras")
            if os.path.exists(single):
                obj.models = [keras.models.load_model(single)]
        return obj

    # ======== внутреннее ========
    def _se_block(self, x, channels: int, ratio: int = 4):
        # Squeeze-Excitation без tf.*: через Keras Reshape + GAP1D
        # x: [batch, channels]
        s = layers.Reshape((1, channels))(x)        # [batch, 1, channels]
        s = layers.GlobalAveragePooling1D()(s)      # [batch, channels]
        s = layers.Dense(max(channels // ratio, 4), activation="relu")(s)
        s = layers.Dense(channels, activation="sigmoid")(s)
        return layers.Multiply()([x, s])            # [batch, channels]

    def _res_block(self, x, units):
        reg = regularizers.l2(self.l2) if self.l2 > 0 else None
        h = layers.Dense(units, kernel_regularizer=reg)(x)
        h = layers.LayerNormalization()(h)
        h = layers.Activation("gelu")(h)
        h = layers.Dropout(self.dropout)(h)
        h = layers.Dense(units, kernel_regularizer=reg)(h)
        h = layers.LayerNormalization()(h)
        h = self._se_block(h, channels=units)  # каналное пере-взвешивание
        x = layers.Add()([x, h])
        x = layers.Activation("gelu")(x)
        return x

    def _build_model(self, input_dim: int, seed: int) -> keras.Model:
        tf.keras.utils.set_random_seed(seed)
        x_in = layers.Input(shape=(input_dim,))
        x = layers.LayerNormalization()(x_in)
        x = layers.Dense(self.hidden_main)(x)
        x = layers.Activation("gelu")(x)
        x = layers.Dropout(self.dropout)(x)
        for _ in range(self.n_resblocks):
            x = self._res_block(x, self.hidden_main)
        x = layers.Dense(self.proj_units)(x)
        x = layers.Activation("gelu")(x)
        x = layers.Dropout(self.dropout)(x)
        out = layers.Dense(1, activation="linear")(x)
        model = keras.Model(x_in, out)
        return model

    def _prepare_X(self, X: Union[pd.DataFrame, np.ndarray, List[Dict]]) -> np.ndarray:
        if isinstance(X, np.ndarray):
            X_arr = X
        elif isinstance(X, list):
            X_arr = np.array([[row.get(f, np.nan) for f in self.feature_order] for row in X], dtype=float)
        else:
            X_arr = X[self.feature_order].astype(float).values
        X_arr = np.nan_to_num(X_arr, nan=0.0)
        return self.scaler.transform(X_arr)

    def _check_ready(self):
        if not self.models or self.scaler is None or self.feature_order is None or \
           self.y_mean_ is None or self.y_std_ is None:
            raise RuntimeError("модель не обучена или не загружена")


# утилита для стабильного сплита (разные сиды → разные фолды ансамбля)
def train_split(X, y, test_size=0.3, seed=42):
    from sklearn.model_selection import StratifiedKFold
    # если регрессия — просто random state сплит:
    from sklearn.model_selection import train_test_split
    return train_test_split(X, y, test_size=test_size, random_state=seed)
