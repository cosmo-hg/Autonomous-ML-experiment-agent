# 🍛 Rebel Foods — Autonomous ML Forecasting Agent

An autonomous ML experiment loop that continuously improves demand forecasting accuracy for Rebel Foods using a **git ratchet** pattern — only improvements get committed.

## 🎯 What it does

- Generates synthetic demand data across **3 kitchens** and **4 SKUs** (Butter Chicken Bowl, Behrouz Biryani, Faasos Rolls, Cheesy Fries)
- Runs automated ML experiments (feature engineering → model training → evaluation)
- Logs every run to `results.tsv` — only commits if MAE improves over the current best
- Visualises all results in a **Streamlit dashboard**

## 📊 Results so far

| Experiment | MAE |
|---|---|
| Baseline (Random Forest) | 31.05 |
| + Lag features (1, 7, 14-day) | 26.55 |
| **+ Rolling stats (7 & 14-day mean/std)** ✅ | **21.40** |
| + Calendar features | 21.65 |
| GradientBoostingRegressor | 23.80 |
| Mean encoding variants | 22–24 |
| Per-entity models | 31.42 |

**Best MAE: 21.3975** — `lag_rolling_7_14` (RandomForestRegressor, 100 estimators)

## 🗂️ Project Structure

```
rebel_forecasting/
├── generate_data.py   # Generates synthetic Rebel Foods demand data
├── prepare.py         # Data loading, train/test split, evaluation (MAE)
├── train.py           # 🤖 Agent edits this — features + model definition
├── dashboard.py       # Streamlit dashboard
├── visualise.py       # Matplotlib chart (results_chart.png)
├── results.tsv        # Full experiment log
├── result.txt         # Current best MAE
├── requirements.txt   # Python dependencies
└── data/
    └── sample_data.csv
```

## 🚀 Run locally

```bash
pip install -r requirements.txt
python generate_data.py         # Generate dataset
python train.py                 # Run an experiment
streamlit run dashboard.py      # Launch dashboard
```

## 🌐 Live Dashboard

Hosted on [Streamlit Community Cloud](https://streamlit.io/cloud).

## ⚙️ How the loop works

```
generate_data → prepare → train → evaluate
                                    ↓
                            MAE improved?
                           /            \
                         YES             NO
                          ↓              ↓
                      git commit      log only, continue
```

The agent follows the [Karpathy AutoResearch](https://github.com/karpathy) pattern — a tight loop of hypothesis → experiment → evaluation → commit gate.

## 📦 Tech Stack

- **Model:** `sklearn.ensemble.RandomForestRegressor`
- **Features:** Lag (1, 7, 14-day) + Rolling mean/std (7 & 14-day window)
- **Dashboard:** Streamlit
- **Data:** Synthetic time-series (seeded for reproducibility)
