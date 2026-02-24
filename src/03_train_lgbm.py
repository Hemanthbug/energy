import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pathlib import Path

DATA = "data/processed/ashrae_sample.parquet"
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_parquet(DATA)

# Target
y = df["y"]

# Features (drop ts; keep for later visualization)
X = df.drop(columns=["y", "ts"])

# Handle categorical
if "primary_use" in X.columns:
    X["primary_use"] = X["primary_use"].astype("category")

# Fill missing numeric values (simple baseline)
for c in X.columns:
    if str(X[c].dtype) != "category":
        X[c] = X[c].fillna(X[c].median())

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train, categorical_feature=["primary_use"] if "primary_use" in X.columns else "auto")
pred = model.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, pred))
print("Validation RMSE (log space):", rmse)

model.booster_.save_model(str(MODEL_DIR / "lgbm.txt"))
print("Saved:", MODEL_DIR / "lgbm.txt")