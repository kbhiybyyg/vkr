import argparse
import json

import joblib
import numpy as np
import os
import pandas as pd

MODELS_DIR='models/properties'
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
    for root, _, files in os.walk(MODELS_DIR):
        for f in files:
            if f.startswith(safe_target) and f.endswith("__meta.json"):
                return os.path.join(root, f)
    raise FileNotFoundError(f"meta.json для '{safe_target}' не найден в {MODELS_DIR}")

def find_model_path(safe_target: str, best_model: str, meta_dir: str) -> str:
    # сначала пробуем рядом с meta.json
    p = os.path.join(meta_dir, f"{safe_target}__{best_model}.joblib")
    if os.path.isfile(p):
        return p
    # иначе ищем рекурсивно
    for root, _, files in os.walk(MODELS_DIR):
        for f in files:
            if f.startswith(safe_target) and f.endswith(".joblib") and best_model in f:
                return os.path.join(root, f)
    raise FileNotFoundError(f"Модель '{best_model}' для '{safe_target}' не найдена")

def predict(df,target):
    safe_target = safe_name(target)
    meta_path=find_meta_path(safe_target)
    with open(meta_path,encoding='utf-8') as f:
        meta=json.load(f)
    best_model=meta.get('best_model')
    model_path=find_model_path(safe_target,best_model,os.path.dirname(meta_path))
    model=joblib.load(model_path)
    features = [c for c in df.columns if c != target]
    df[target] = model.predict(df[features])
    return df

DEFAULT_TARGETS = [
    "Модуль упругости при растяжении, ГПа",
    "Прочность при растяжении, МПа",
]

def main():
    ap = argparse.ArgumentParser(description="Простой CLI для прогнозов свойств (sklearn)")
    ap.add_argument("--input", required=True, help="входной CSV с признаками")
    ap.add_argument("--output", default="", help="выходной CSV; если не указан — печать в stdout")
    ap.add_argument("--target", default="", help="конкретный таргет; если пусто — считаем оба")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df = to_numeric(df)

    targets = [args.target] if args.target else DEFAULT_TARGETS
    for t in targets:
        df = predict(df, t)
        print(f"[ok] предсказано: {t}")

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Сохранено: {args.output}")
    else:
        print(df.to_csv(index=False))

if __name__ == "__main__":
    main()