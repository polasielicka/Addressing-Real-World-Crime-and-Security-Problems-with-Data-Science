import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet

DATA_DIR = "../data_CBL"

def load_burglary_counts(data_dir):
    street_files = glob.glob(os.path.join(data_dir, "*", "*-street.csv"))
    df = pd.concat((pd.read_csv(f) for f in street_files), ignore_index=True)
    df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
    df = df[df["Crime type"].str.strip().str.lower() == "burglary"]
    monthly_counts = df.set_index("Month").resample("M").size()
    return monthly_counts.asfreq("M").fillna(0)

def forecast_expsmoothing(train):
    model = ExponentialSmoothing(train, trend="add", seasonal="add", seasonal_periods=12)
    fit = model.fit()
    return fit.forecast(6)

def forecast_sarima(train):
    model = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12))
    fit = model.fit(disp=False)
    return fit.forecast(6)

def forecast_prophet(train):
    df_prophet = train.reset_index()
    df_prophet.columns = ["ds", "y"]
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=6, freq="M")
    forecast = model.predict(future)
    forecast.set_index("ds", inplace=True)
    return forecast["yhat"][-6:]

def evaluate_model(name, forecast, actual):
    mae = mean_absolute_error(actual, forecast)
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    print(f"{name} Forecast Accuracy:")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

def plot_comparison(actual, forecasts):
    plt.figure(figsize=(14, 6))
    actual.plot(label="Actual", marker="o", color="black")
    for name, forecast in forecasts.items():
        forecast.plot(label=name, marker="x")
    plt.title("Forecast Comparison (Last 6 Months)")
    plt.xlabel("Month")
    plt.ylabel("Number of Burglaries")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_full_continuation(actual, forecasts):
    plt.figure(figsize=(14, 6))

    # Plot full actuals
    actual.plot(label="Actual", color="black", marker="o")

    # Extend each model forecast from the last actual month
    last_month = actual.index[-1]
    for name, forecast in forecasts.items():
        forecast_series = pd.Series(forecast.values, index=pd.date_range(start=last_month + pd.offsets.MonthBegin(1),
                                                                         periods=len(forecast), freq='MS'))
        forecast_series.plot(label=f"{name} (forecast)", marker="x")

    plt.title("Full Series with Forecast Continuation (Next 6 Months)")
    plt.xlabel("Month")
    plt.ylabel("Number of Burglaries")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    monthly_counts = load_burglary_counts(DATA_DIR)

    train = monthly_counts[:-6]
    test = monthly_counts[-6:]

    forecasts = {
        "Exponential Smoothing": forecast_expsmoothing(train),
        "SARIMA": forecast_sarima(train),
        "Prophet": forecast_prophet(train)
    }

    plot_comparison(test, forecasts)

    print("\n🧾 Last 6 months of actual data:\n")
    print(test)

    for name, forecast in forecasts.items():
        print(f"\n{name} forecast:\n{forecast}")
        evaluate_model(name, forecast, test)

    plot_full_continuation(monthly_counts, forecasts)

if __name__ == "__main__":
    main()
