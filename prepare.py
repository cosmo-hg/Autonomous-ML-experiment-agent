import pandas as pd
from sklearn.metrics import mean_absolute_error


def load_raw() -> pd.DataFrame:
    df = pd.read_csv("data/sample_data.csv", parse_dates=["date"])
    return df.sort_values(["kitchen_id", "sku", "date"]).reset_index(drop=True)


def temporal_split(df: pd.DataFrame, test_days: int = 18):
    cutoff = df["date"].max() - pd.Timedelta(days=test_days)
    train  = df[df["date"] <= cutoff].copy()
    test   = df[df["date"] >  cutoff].copy()
    return train, test


def evaluate(y_true, y_pred) -> float:
    return mean_absolute_error(y_true, y_pred)
