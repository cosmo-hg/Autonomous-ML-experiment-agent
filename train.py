import os
import csv
import pandas as pd
from prepare import load_raw, temporal_split, evaluate

# ── AGENT EDITS THIS BLOCK ────────────────────────────────────────────────────

EXPERIMENT_NAME = "baseline_random_forest"

def build_features(df: pd.DataFrame, train_means=None):
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"]  = (df["day_of_week"] >= 5).astype(int)
    df = pd.get_dummies(df, columns=["kitchen_id", "sku"])
    return df, train_means

from sklearn.ensemble import RandomForestRegressor

def build_model():
    return RandomForestRegressor(n_estimators=100, random_state=42)

# ── DO NOT EDIT BELOW THIS LINE ───────────────────────────────────────────────

raw                 = load_raw()
train_raw, test_raw = temporal_split(raw)

train, means = build_features(train_raw)
test,  _     = build_features(test_raw, train_means=means)

train_cols = [c for c in train.columns if c not in ("date", "demand")]
test       = test.reindex(columns=["date", "demand"] + train_cols, fill_value=0)

X_train, y_train = train[train_cols], train["demand"]
X_test,  y_test  = test[train_cols],  test["demand"]

model = build_model()
model.fit(X_train, y_train)
pred  = model.predict(X_test)
mae   = evaluate(y_test, pred)

print(f"Experiment : {EXPERIMENT_NAME}")
print(f"MAE        : {mae:.4f}")

log_path     = "results.tsv"
write_header = not os.path.exists(log_path)
with open(log_path, "a", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    if write_header:
        w.writerow(["experiment", "mae"])
    w.writerow([EXPERIMENT_NAME, round(mae, 4)])

with open("result.txt", "w") as f:
    f.write(str(round(mae, 4)))
