import argparse

import numpy as np
import os
import json
import pickle
import keras
import pandas as pd
MODELS_DIR='models/properties'
TARGET_COLUMS='Соотношение матрица-наполнитель'
def to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    for c in df2.columns:
        if df2[c].dtype == object:
            df2[c] = pd.to_numeric(df2[c].astype(str).str.replace(",", "."), errors="coerce")
    return df2

def load_ratio_system(model_dir: str):
    meta_path  = os.path.join(model_dir, "meta.json")
    scaler_pkl = os.path.join(model_dir, "scaler.pkl")
    model_kr   = os.path.join(model_dir, "keras_model.keras")

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
def predict_ratio_df(df: pd.DataFrame, model_dir: str) -> pd.DataFrame:
    model, scaler, feature_order, clip = load_ratio_system(MODELS_DIR)

    missing = [c for c in feature_order if c not in df.columns]
    if missing:
        raise ValueError(f"в данных нет нужных колонок: {missing}")

    X = df[feature_order].astype(float).values
    Xs = scaler.transform(np.nan_to_num(X, nan=0.0))

    y = model.predict(Xs, verbose=0).reshape(-1)
    y = np.clip(y, *clip)

    out = df.copy()
    out[TARGET_COLUMS] = y
    return out

def read_table(path: str) -> pd.DataFrame:
    p = path.lower()
    if p.endswith(".csv"):
        return pd.read_csv(path)
    elif p.endswith(".jsonl") or p.endswith(".json"):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    else:
        raise ValueError("поддерживаются только .csv, .json, .jsonl")

def main():
    ap = argparse.ArgumentParser(description="CLI: рекомендация соотношения матрица-наполнитель (TF)")
    ap.add_argument("--input", required=True, help="вход: CSV или JSON/JSONL")
    ap.add_argument("--output", default="", help="выход: CSV/JSON (если пусто — печать в stdout)")
    ap.add_argument("--model-dir", default=MODELS_DIR, help="папка модели (по умолчанию models/ratio)")
    ap.add_argument("--print-schema", action="store_true", help="показать ожидаемые признаки и выйти")
    args = ap.parse_args()


    df = read_table(args.input)
    df = to_numeric(df)

    out = predict_ratio_df(df, args.model_dir)

    if args.output:
        if args.output.lower().endswith(".csv"):
            out.to_csv(args.output, index=False)
        else:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(out.to_json(orient="records", ensure_ascii=False, indent=2))
        print(f"OK: сохранено в {args.output}")
    else:
        # печатаем CSV в stdout
        print(out.to_csv(index=False))

if __name__ == "__main__":
    main()
