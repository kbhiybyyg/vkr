import json
import pickle

import joblib
import keras
import numpy as np
import os
import pandas as pd

MODELS_DIR_RATIO = 'models/ratio'
MODELS_DIR_PROPERTIES = 'models/properties'


def safe_name(s: str) -> str:
    return (s.replace(" ", "_")
            .replace(",", "")
            .replace(":", "")
            .replace("(", "")
            .replace(")", ""))


def to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for c in df2.columns:
        if df2[c].dtype == object:
            df2[c] = pd.to_numeric(df2[c].astype(str).str.replace(",", "."), errors="coerce")
    return df2


def find_meta_path(safe_target: str) -> str:
    # ищем meta-файл рекурсивно внутри models/properties
    for root, _, files in os.walk(MODELS_DIR_PROPERTIES):
        for f in files:
            if f.startswith(safe_target) and f.endswith("__meta.json"):
                return os.path.join(root, f)
    raise FileNotFoundError(f"meta.json для '{safe_target}' не найден в {MODELS_DIR_PROPERTIES}")


def find_model_path(safe_target: str, best_model: str, meta_dir: str) -> str:
    # сначала пробуем рядом с meta.json
    p = os.path.join(meta_dir, f"{safe_target}__{best_model}.joblib")
    if os.path.isfile(p):
        return p
    # иначе ищем рекурсивно
    for root, _, files in os.walk(MODELS_DIR_PROPERTIES):
        for f in files:
            if f.startswith(safe_target) and f.endswith(".joblib") and best_model in f:
                return os.path.join(root, f)
    raise FileNotFoundError(f"Модель '{best_model}' для '{safe_target}' не найдена")


def predict(df, target):
    safe_target = safe_name(target)
    meta_path = find_meta_path(safe_target)
    with open(meta_path, encoding='utf-8') as f:
        meta = json.load(f)
    best_model = meta.get('best_model')
    model_path = find_model_path(safe_target, best_model, os.path.dirname(meta_path))
    model = joblib.load(model_path)
    features = [c for c in df.columns if c != target]
    df[target] = model.predict(df[features])
    return df


def load_ratio_system(model_dir: str):
    meta_path = os.path.join(model_dir, "meta.json")
    scaler_pkl = os.path.join(model_dir, "scaler.pkl")
    model_kr = os.path.join(model_dir, "keras_model.keras")

    if not (os.path.isfile(meta_path) and os.path.isfile(scaler_pkl) and os.path.isfile(model_kr)):
        raise FileNotFoundError(f"в {model_dir} нужны meta.json, scaler.pkl и keras_model.keras")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    with open(scaler_pkl, "rb") as f:
        scaler = pickle.load(f)
    model = keras.models.load_model(model_kr)

    feature_order = meta.get("feature_order")
    if not feature_order:
        raise ValueError("в meta.json нет 'feature_order'")
    clip = tuple(meta.get("hyperparams", {}).get("clip_output", [0.0, 1.0]))
    return model, scaler, feature_order, clip


def predict_ratio(df: pd.DataFrame) -> pd.DataFrame:
    model, scaler, feature_order, clip = load_ratio_system(MODELS_DIR_RATIO)

    missing = [c for c in feature_order if c not in df.columns]
    if missing:
        raise ValueError(f"в данных нет нужных колонок: {missing}")

    X = df[feature_order].astype(float).values
    Xs = scaler.transform(np.nan_to_num(X, nan=0.0))

    y = model.predict(Xs, verbose=0).reshape(-1)

    out = df.copy()
    out['Соотношение матрица-наполнитель'] = y
    return out
