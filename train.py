import os
import csv
import pandas as pd
from prepare import load_raw, temporal_split, evaluate
from sklearn.ensemble import RandomForestRegressor

# ── AGENT EDITS THIS BLOCK ────────────────────────────────────────────────────

EXPERIMENT_NAME = "lag_rolling_7_14"

def build_features(df: pd.DataFrame, train_means=None):
    df = df.copy()
    df = df.sort_values(["kitchen_id", "sku", "date"])

    df["lag_7"]  = df.groupby(["kitchen_id", "sku"])["demand"].shift(7)
    df["lag_14"] = df.groupby(["kitchen_id", "sku"])["demand"].shift(14)
    df["roll_mean_7"] = df.groupby(["kitchen_id", "sku"])["demand"].transform(
        lambda x: x.shift(1).rolling(7).mean())
    df["roll_std_7"] = df.groupby(["kitchen_id", "sku"])["demand"].transform(
        lambda x: x.shift(1).rolling(7).std())
    df["roll_mean_14"] = df.groupby(["kitchen_id", "sku"])["demand"].transform(
        lambda x: x.shift(1).rolling(14).mean())
    df["roll_std_14"] = df.groupby(["kitchen_id", "sku"])["demand"].transform(
        lambda x: x.shift(1).rolling(14).std())

    df = df.dropna()
    df = pd.get_dummies(df, columns=["kitchen_id", "sku"])
    return df, None

def build_model():
    return RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

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
