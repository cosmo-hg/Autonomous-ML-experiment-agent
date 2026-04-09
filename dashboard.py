import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from prepare import load_raw, temporal_split, evaluate
from train import build_features, build_model

st.set_page_config(
    page_title="Rebel Foods Demand Forecasting",
    page_icon="🍛",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────────────────────

st.title("🍛 Rebel Foods — Daily Demand Forecasting")
st.caption("An autonomous ML experiment loop that continuously improves forecasting accuracy using a git ratchet to keep only improvements.")
st.divider()

# ── Load results ──────────────────────────────────────────────────────────────

results = pd.read_csv("results.tsv", sep="\t")
best_mae = results["mae"].min()
baseline_mae = results["mae"].iloc[0]
improvement = round(((baseline_mae - best_mae) / baseline_mae) * 100, 1)
best_experiment = results.loc[results["mae"].idxmin(), "experiment"]
total_experiments = len(results)

# ── Metric cards ─────────────────────────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)
col1.metric("Baseline MAE", f"{baseline_mae:.2f}")
col2.metric("Best MAE", f"{best_mae:.2f}", delta=f"-{baseline_mae - best_mae:.2f}")
col3.metric("Improvement", f"{improvement}%")
col4.metric("Experiments Run", total_experiments)

st.divider()

# ── Layout: two columns ───────────────────────────────────────────────────────

left, right = st.columns(2)

# ── MAE bar chart ─────────────────────────────────────────────────────────────

with left:
    st.subheader("MAE across experiments")
    st.caption("Green bars were committed. Red bars did not improve over the current best and were rejected.")

    colors = []
    for i, row in results.iterrows():
        if i == 0:
            colors.append("#5B9BD5")
        elif row["mae"] < results.loc[:i-1, "mae"].min():
            colors.append("#4CAF50")
        else:
            colors.append("#E57373")

    fig1, ax1 = plt.subplots(figsize=(8, 4))
    bars = ax1.bar(results["experiment"], results["mae"], color=colors, edgecolor="none")
    ax1.axhline(baseline_mae, color="#E57373", linestyle="--", linewidth=1.2, label=f"Baseline ({baseline_mae:.2f})")
    ax1.axhline(best_mae, color="#4CAF50", linestyle="--", linewidth=1.2, label=f"Best ({best_mae:.2f})")
    ax1.set_ylabel("MAE (lower is better)")
    ax1.set_xlabel("")
    ax1.tick_params(axis="x", rotation=45, labelsize=7)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    green_patch = mpatches.Patch(color="#4CAF50", label="Committed improvement")
    red_patch   = mpatches.Patch(color="#E57373", label="Rejected")
    blue_patch  = mpatches.Patch(color="#5B9BD5", label="Baseline")
    ax1.legend(handles=[blue_patch, green_patch, red_patch], fontsize=8)

    plt.tight_layout()
    st.pyplot(fig1)

# ── Actual vs predicted ───────────────────────────────────────────────────────

with right:
    st.subheader("Actual vs predicted demand")

    raw                 = load_raw()
    train_raw, test_raw = temporal_split(raw)
    train, means        = build_features(train_raw)
    test,  _            = build_features(test_raw, train_means=means)

    train_cols = [c for c in train.columns if c not in ("date", "demand")]
    test       = test.reindex(columns=["date", "demand"] + train_cols, fill_value=0)

    model = build_model()
    model.fit(train[train_cols], train["demand"])
    pred  = model.predict(test[train_cols])

    test_raw             = test_raw.copy()
    test_raw["predicted"] = pd.Series(pred, index=test.index) # FIXED: Map by index to avoid length issues

    kitchens = sorted(test_raw["kitchen_id"].unique())
    skus     = sorted(test_raw["sku"].unique())

    selected_kitchen = st.selectbox("Kitchen", kitchens)
    selected_sku     = st.selectbox("SKU", skus)

    pair = test_raw[
        (test_raw["kitchen_id"] == selected_kitchen) &
        (test_raw["sku"]        == selected_sku)
    ]

    # Evaluate ignores NaNs via mapping above if safe, but pred is shorter.
    # To be perfectly safe for pairwise evaluation, drop prediction NaNs (the dropped lag rows).
    pair_eval = pair.dropna(subset=["predicted"])
    if not pair_eval.empty:
        pair_mae = evaluate(pair_eval["demand"], pair_eval["predicted"])
    else:
        pair_mae = 0.0

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.plot(pair["date"].values, pair["demand"].values,    label="Actual",    marker="o", markersize=4, linewidth=1.5, color="#5B9BD5") # FIXED: .values
    ax2.plot(pair["date"].values, pair["predicted"].values, label="Predicted", marker="x", markersize=4, linewidth=1.5, color="#FF8C42", linestyle="--") # FIXED: .values
    ax2.set_ylabel("Demand (units)")
    ax2.set_xlabel("")
    ax2.tick_params(axis="x", rotation=30, labelsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend(fontsize=9)
    ax2.set_title(f"MAE for this pair: {pair_mae:.2f}", fontsize=9, color="gray")
    plt.tight_layout()
    st.pyplot(fig2)

st.divider()

# ── Experiment log table ──────────────────────────────────────────────────────

st.subheader("Full experiment log")
st.caption("All 22 experiments run by the agent. Only improvements were committed to git.")

styled = results.copy()
styled["status"] = styled["mae"].apply(
    lambda x: "Best" if x == best_mae else ("Baseline" if x == baseline_mae else ("Committed" if x < baseline_mae else "Rejected"))
)
styled["mae"] = styled["mae"].apply(lambda x: f"{x:.4f}")

st.dataframe(
    styled,
    use_container_width=True,
    hide_index=True,
    column_config={
        "experiment": st.column_config.TextColumn("Experiment", width="large"),
        "mae":        st.column_config.TextColumn("MAE"),
        "status":     st.column_config.TextColumn("Status"),
    }
)

# ── Footer ────────────────────────────────────────────────────────────────────

st.divider()
st.caption("Autonomous experiment loop built from scratch. Model: RandomForestRegressor with lag and rolling features. Dataset: Rebel Foods synthetic data across 3 kitchens and 4 SKUs.")
