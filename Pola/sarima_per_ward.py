import os
import glob
import csv
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from pmdarima import auto_arima
except ImportError as err:
    raise ImportError("pmdarima is required – install via `pip install pmdarima`.") from err

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Config
DATA_DIR = "../data_all"
FORECAST_HORIZON = 24
FUTURE_PERIODS = 12
SEASONAL_PERIOD = 12


def lsoa_mapping():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data_all')  # <-- corrected path
    combined_data = glob.glob(os.path.join(data_dir, '**', '*-street.csv'), recursive=True)

    if not combined_data:
        raise FileNotFoundError(f"No CSVs found in {data_dir}")

    lookup_path_2024 = os.path.join(script_dir, '..', 'data', 'best_fit_lsoa_data (dont use)',
                                    'LSOA_(2021)_to_Electoral_Ward_(2024)_to_LAD_(2024)_Best_Fit_Lookup_in_EW.csv')
    lookup_path_2018 = os.path.join(script_dir, '..', 'data', 'best_fit_lsoa_data (dont use)',
                                    'Lower_Layer_Super_Output_Area_(2011)_to_Ward_(2018)_Lookup_in_England_and_Wales_v3.csv')

    df_lsoa = pd.concat((pd.read_csv(f) for f in combined_data), ignore_index=True)

    lsoa_lookup = pd.read_csv(lookup_path_2024)
    lsoa_lookup2 = pd.read_csv(lookup_path_2018)

    df1 = df_lsoa.merge(
        lsoa_lookup[['LSOA21CD', 'WD24NM']],
        how='left',
        left_on='LSOA code',
        right_on='LSOA21CD'
    ).rename(columns={'WD24NM': 'ward2024'}).drop(columns=['LSOA21CD'])

    df = df1.merge(
        lsoa_lookup2[['LSOA11CD', 'WD18NM']],
        how='left',
        left_on='LSOA code',
        right_on='LSOA11CD'
    ).rename(columns={'WD18NM': 'ward2018'}).drop(columns=['LSOA11CD'])

    df['ward_name'] = df['ward2024'].fillna(df['ward2018'])
    burglary_df = df[df["Crime type"].str.lower() == "burglary"]
    return burglary_df[burglary_df['LSOA code'].notna()]


def load_burglary_counts_by_ward() -> pd.DataFrame:
    df = lsoa_mapping()
    df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
    grouped = (
        df.groupby(["ward_name", "Month"])
          .size()
          .unstack(fill_value=0)
          .T
    )
    return grouped

def train_test_split(series: pd.Series, horizon: int) -> Tuple[pd.Series, pd.Series]:
    test_index = series.index[-horizon:]
    test = series.loc[test_index]
    train = series.loc[:test_index[0] - pd.offsets.MonthBegin()]
    return train, test

def fit_sarima(train: pd.Series, seasonal=True):
    return auto_arima(
        train,
        seasonal=seasonal,
        m=SEASONAL_PERIOD if seasonal else 1,
        stepwise=True,
        trace=False,
        error_action="ignore",
        suppress_warnings=True,
        max_order=10,
        information_criterion="aicc",
    )

def evaluate(y_true: pd.Series, y_pred: pd.Series) -> Tuple[float, float, float, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    smape = 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-8))
    naive = y_true.shift(SEASONAL_PERIOD)
    mase_denom = np.mean(np.abs(y_true[SEASONAL_PERIOD:] - naive[SEASONAL_PERIOD:]))
    mase = mae / mase_denom if mase_denom != 0 else np.nan
    return mae, rmse, smape, mase

def main():
    print("Loading burglary data by ward...")
    ward_monthly = load_burglary_counts_by_ward()
    print(f"Number of wards being forecasted: {ward_monthly.shape[1]}")

    results = []

    for ward in ward_monthly.columns:
        print(f"\n=== Forecasting for ward: {ward} ===")
        series = ward_monthly[ward].asfreq("MS")

        if series.dropna().shape[0] < 12:
            print("Very short series – model may be weak, but proceeding...")

        try:
            train, test = train_test_split(series, FORECAST_HORIZON)

            try:
                sarima = fit_sarima(train, seasonal=True)
            except:
                print("Seasonal model failed, trying non-seasonal ARIMA.")
                sarima = fit_sarima(train, seasonal=False)

            forecast = pd.Series(sarima.predict(n_periods=len(test)), index=test.index)
            mae, rmse, smape, mase = evaluate(test, forecast)

            print(f"MAE: {mae:.2f}  RMSE: {rmse:.2f}  SMAPE: {smape:.2f}%  MASE: {mase:.2f}")

            results.append({
                "ward": ward,
                "mae": mae,
                "rmse": rmse,
                "smape": smape,
                "mase": mase,
            })

        except Exception as e:
            print(f"Failed to model {ward}: {e}")
            results.append({
                "ward": ward,
                "mae": None,
                "rmse": None,
                "smape": None,
                "mase": None,
                "error": str(e),
            })

    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", "ward_forecast_metrics.csv")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\n Forecast metrics saved to: {output_path}")

if __name__ == "__main__":
    main()
