import os
import glob
import argparse
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.statespace.sarimax import SARIMAX

DATA_DIR = "../data_all"
FORECAST_HORIZON = 24             # months kept for out‑of‑sample evaluation
FUTURE_PERIODS = 12               # months to forecast after refitting on full data
SEASONAL_PERIOD = 12              # monthly seasonality (12 = yearly)

def load_burglary_counts(data_dir: str) -> pd.Series:
    pattern = os.path.join(data_dir, "*", "*", "*-street.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No street CSVs found using pattern: {pattern}")

    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")

    df = df[df["Crime type"].str.strip().str.lower() == "burglary"]

    monthly = (
        df.set_index("Month")
          .resample("MS")
          .size()
          .asfreq("MS")
          .fillna(0)
    )
    return monthly

def train_test_split(series: pd.Series, horizon: int) -> Tuple[pd.Series, pd.Series]:
    """Return (train, test) where *test* covers the last *horizon* months."""
    test = series.last(f"{horizon}MS")
    train = series.loc[: test.index[0] - pd.offsets.MonthBegin()]
    return train, test

def fit_sarima(train: pd.Series):
    """Fit SARIMA(1,1,1)(1,0,2,12) model to training data."""
    model = SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 2, SEASONAL_PERIOD),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    return results

def evaluate(y_true: pd.Series, y_pred: pd.Series) -> Tuple[float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return mae, rmse

def plot_series(history: pd.Series, test: pd.Series, forecast: pd.Series, title: str):
    plt.figure(figsize=(14, 6))
    history.plot(label="Train", color="black")
    test.plot(label="Actual (hold‑out)", color="tab:blue")
    forecast.plot(label="SARIMA forecast", color="tab:orange")
    plt.axvline(test.index[0], color="grey", ls="--", alpha=0.6)
    plt.title(title)
    plt.xlabel("Month")
    plt.ylabel("Number of Burglaries")
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------
# Main routine
# -----------------------------

def main(data_dir: str = DATA_DIR):
    # 1. Load data
    series = load_burglary_counts(data_dir)
    print(f"Loaded {len(series)} monthly observations (from {series.index[0].date()} to {series.index[-1].date()})")

    # 2. Train‑test split
    train, test = train_test_split(series, FORECAST_HORIZON)
    print(f"Training on {len(train)} months, testing on {len(test)} months")

    # 3. Fit SARIMA on training set
    print("\n>>> Fitting SARIMA(1,1,1)(1,0,2,12) model...")
    sarima = fit_sarima(train)
    print(f"\nModel fitted. AIC: {sarima.aic:.2f}")

    forecast_values = sarima.predict(start=test.index[0], end=test.index[-1])
    forecast = pd.Series(forecast_values, index=test.index, name="forecast")

    mae, rmse = evaluate(test, forecast)
    print(f"\nHold‑out performance  (horizon = {len(test)} months)\nMAE : {mae:,.2f}\nRMSE: {rmse:,.2f}")

    # 5. Plot comparison
    plot_series(train, test, forecast, "SARIMA Hold‑out Forecast vs Actual")

    # 6. Re‑fit on full data and forecast the future
    sarima = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 2, SEASONAL_PERIOD),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    future_index = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(), periods=FUTURE_PERIODS, freq="MS")
    future_forecast_values = sarima.predict(start=len(series), end=len(series) + FUTURE_PERIODS - 1)
    future_forecast = pd.Series(future_forecast_values, index=future_index, name="future_forecast")

    print("\nNext 12‑month forecast:")
    print(future_forecast)

    # Plot forecasted values as bar chart
    plt.figure(figsize=(12, 6))
    future_forecast.plot(kind='bar', color='skyblue')
    plt.title("Forecasted Burglaries for Next 12 Months (SARIMA)")
    plt.xlabel("Month")
    plt.ylabel("Predicted Number of Burglaries")
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Plot full series with forecast
    plt.figure(figsize=(14, 6))
    series.plot(label="History", color="black")
    future_forecast.plot(label="12‑month forecast", color="tab:red")
    plt.title("SARIMA Forecast – Next 12 Months")
    plt.xlabel("Month")
    plt.ylabel("Number of Burglaries")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SARIMA forecasting for burglary data")
    parser.add_argument("--data", "-d", default=DATA_DIR, help="Root folder of CSV files (default: ../data_all)")
    args = parser.parse_args()
    main(args.data)
