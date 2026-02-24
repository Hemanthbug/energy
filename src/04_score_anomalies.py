import pandas as pd
import numpy as np
import lightgbm as lgb
from pathlib import Path

DATA = "data/processed/ashrae_sample.parquet"
MODEL_PATH = "models/lgbm.txt"
OUT_PATH = "data/processed/anomaly_scored.parquet"

df = pd.read_parquet(DATA)

y_true = df["y"].values
X = df.drop(columns=["y"])

# Keep timestamp for charts
ts = X["ts"]
X = X.drop(columns=["ts"])

if "primary_use" in X.columns:
    X["primary_use"] = X["primary_use"].astype("category")

for c in X.columns:
    if str(X[c].dtype) != "category":
        X[c] = X[c].fillna(X[c].median())

model = lgb.Booster(model_file=MODEL_PATH)
y_pred = model.predict(X)

df_out = df.copy()
df_out["y_pred"] = y_pred
df_out["residual"] = df_out["y"] - df_out["y_pred"]

# Simple anomaly rule: top 0.5% residual magnitude
thr = df_out["residual"].abs().quantile(0.995)
df_out["is_anomaly"] = (df_out["residual"].abs() >= thr).astype(int)

df_out.to_parquet(OUT_PATH, index=False)
print("Saved:", OUT_PATH, "| anomaly threshold:", thr)