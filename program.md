# Rebel Foods Demand Forecasting — Research Goal

Metric:  Mean Absolute Error (MAE) on a held-out temporal test set.
         Lower is better. Commit only when MAE improves.

Baseline: 21.3975 (lag_rolling_7_14)

Constraints:
- Each experiment must complete in under 8 minutes wall-clock time.
- Only edit train.py. Do not touch prepare.py or the data files.
- Features must be derived from: date, kitchen_id, sku, demand history.
- Log every run to results.tsv whether it improves or not.
- Use git to commit only runs that beat the current best MAE.
- Never stop the loop. If an experiment fails, log the error and continue.

Suggested directions (in rough priority order):
1. Lag features: demand at t-1, t-7, t-14 per kitchen/SKU series
2. Rolling statistics: 7-day and 14-day rolling mean and std
3. Calendar features: day_of_week, is_weekend, week_of_year
4. Mean-encode kitchen_id and sku on the training split only
5. Model selection: try GradientBoostingRegressor, LightGBM, XGBoost
6. Hyperparameter search: n_estimators, max_depth, learning_rate
