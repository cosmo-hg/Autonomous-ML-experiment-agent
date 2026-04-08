import pandas as pd
import matplotlib.pyplot as plt
from prepare import load_raw, temporal_split
from train import build_features, build_model, train_cols

results = pd.read_csv("results.tsv", sep="\t")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(results["experiment"], results["mae"], color="steelblue")
axes[0].set_title("MAE per experiment")
axes[0].set_xlabel("Experiment")
axes[0].set_ylabel("MAE")
axes[0].tick_params(axis="x", rotation=45)
axes[0].axhline(results["mae"].iloc[0], color="red", linestyle="--", label="baseline")
axes[0].legend()

raw                 = load_raw()
train_raw, test_raw = temporal_split(raw)
train, means        = build_features(train_raw)
test,  _            = build_features(test_raw, train_means=means)
test                = test.reindex(columns=["date", "demand"] + train_cols, fill_value=0)

X_train, y_train = train[train_cols], train["demand"]
X_test,  y_test  = test[train_cols],  test["demand"]

model = build_model()
model.fit(X_train, y_train)
pred = model.predict(X_test)

sample              = test_raw.copy()
sample["predicted"] = pred
pair = sample[
    (sample["kitchen_id"] == "Mumbai_Kitchen_1") &
    (sample["sku"]        == "Butter_Chicken_Bowl")
]

axes[1].plot(pair["date"], pair["demand"],    label="Actual",    marker="o", markersize=3)
axes[1].plot(pair["date"], pair["predicted"], label="Predicted", marker="x", markersize=3)
axes[1].set_title("Actual vs predicted — Mumbai / Butter Chicken Bowl")
axes[1].set_xlabel("Date")
axes[1].set_ylabel("Demand")
axes[1].legend()
axes[1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("results_chart.png", dpi=150)
plt.show()
print("Chart saved to results_chart.png")
