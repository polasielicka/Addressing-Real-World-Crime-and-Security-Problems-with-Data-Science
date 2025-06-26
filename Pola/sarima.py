import os
import glob
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

DATA_DIR = "../data_all"
OUTPUT_PATH = "../output/sarima_forecast.csv"
FUTURE_PERIODS = 12  # months to forecast
SEASONAL_PERIOD = 12

def load_burglary_counts(data_dir: str) -> pd.Series:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.abspath(os.path.join(script_dir, '..', 'output'))

    # read input burglary data from csv
    all_data_path = os.path.join(output_dir, 'input_data.csv')
    if not os.path.exists(all_data_path):
        raise FileNotFoundError(f"'input_data.csv' not found at {all_data_path}. Please ensure the file exists.")
    df = pd.read_csv(all_data_path) # potentially clunky

    df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")

    monthly = (
        df.set_index("Month")
          .resample("MS")
          .size()
          .asfreq("MS")
          .fillna(0)
    )
    return monthly

def main():
    series = load_burglary_counts(DATA_DIR)

    # Fit SARIMA on full data
    model = SARIMAX(
        series,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 2, SEASONAL_PERIOD),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)

    # Forecast next 12 months
    future_index = pd.date_range(series.index[-1] + pd.offsets.MonthBegin(), periods=FUTURE_PERIODS, freq="MS")
    forecast_values = model.predict(start=len(series), end=len(series) + FUTURE_PERIODS - 1)

    forecast_df = pd.DataFrame({
        "month": future_index,
        "forecast": forecast_values
    })

    # Save to CSV
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    forecast_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Forecast saved to: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
