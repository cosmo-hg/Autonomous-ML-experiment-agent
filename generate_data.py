import pandas as pd
import numpy as np

np.random.seed(42)

KITCHENS = ["Mumbai_Kitchen_1", "Delhi_Kitchen_1", "Bangalore_Kitchen_1"]
SKUS     = ["Butter_Chicken_Bowl", "Behrouz_Biryani", "Faasos_Pizza", "Cheesy_Fries"]
DATES    = pd.date_range("2025-01-01", periods=90, freq="D")

records = []
for kitchen in KITCHENS:
    for sku in SKUS:
        base      = np.random.randint(80, 300)
        noise_std = np.random.uniform(10, 40)
        for i, date in enumerate(DATES):
            trend          = i * 0.4
            weekend_factor = 1.35 if date.dayofweek >= 5 else 1.0
            day_sin        = 10 * np.sin(2 * np.pi * date.dayofweek / 7)
            noise          = np.random.normal(0, noise_std)
            demand         = max(0, int((base + trend + day_sin) * weekend_factor + noise))
            records.append([kitchen, sku, date, demand])

df = pd.DataFrame(records, columns=["kitchen_id", "sku", "date", "demand"])
df.to_csv("data/sample_data.csv", index=False)
print(f"Dataset saved. Shape: {df.shape}")
print(df.head())
