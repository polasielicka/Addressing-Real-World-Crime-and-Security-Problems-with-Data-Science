import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error, r2_score
import os

def load_data():
    return pd.read_csv("output/data_cleaned.csv")

def add_lags(df, lag_range):  # lag_1 to lag_12
    df = df.sort_values(["ward_name", "year", "month_num"])
    for lag in lag_range:
        df[f"lag_{lag}"] = df.groupby("ward_name")["burglaries"].shift(lag)
    return df

def add_covid_flag(df):
    # exclude first 2 months of 2020
    df["covid_flag"] = ((df["year"] > 2020) | ((df["year"] == 2020) & (df["month_num"] > 2))).astype(int)
    return df

def recursive_forecast(df, forecast_years, lag_range):
    df = add_lags(df, lag_range)
    df = add_covid_flag(df)
    
    df_results_list = []
    model = None

    features = [
        "imd_score", "income_score", "employment_domain_score",
        "education_domain_score", "health_domain_score", "crime_domain_score",
        "housing_domain_score", "environment_domain_score",
        "covid_flag", "month_num"
    ] + [f"lag_{i}" for i in lag_range]

    # collect true and predicted values to evaluate later
    y_true_all = []
    y_pred_all = []

    for year in forecast_years:
        train_df = df[df["year"] < year].copy()
        test_df = df[df["year"] == year].copy()

        train_df = train_df.dropna(subset=["lag_1", "lag_2", "lag_3"])

        X_train = train_df[features]
        y_train = train_df["burglaries"]

        model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
        model.fit(X_train, y_train)

        for month in lag_range:
            month_df = test_df[test_df["month_num"] == month].copy()
            preds = []

            for idx, row in month_df.iterrows():
                ward = row["ward_name"]
                past_data = df[(df["ward_name"] == ward) & (
                    (df["year"] < year) | ((df["year"] == year) & (df["month_num"] < month))
                )].sort_values(["year", "month_num"]).tail(12)

                if len(past_data) < 3:
                    # Skip if not enough past data to fill lags
                    continue

                lags = past_data["burglaries"].values[::-1]  # latest to oldest
                if len(lags) < len(lag_range):
                    continue
                for i in range(len(lag_range)):
                    row[f"lag_{i+1}"] = lags[i]

                row["lag_1"], row["lag_2"], row["lag_3"] = lags[0], lags[1], lags[2]

                input_row = row[features].values.reshape(1, -1)
                pred = model.predict(input_row)[0]

                preds.append({
                    "ward_name": ward,
                    "month_num": month,
                    "predicted_burglaries": pred,
                    "actual_burglaries": row["burglaries"]
                })

                # Add prediction to df for future lag use
                new_row = row.copy()
                new_row["burglaries"] = pred
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

                # For evaluation
                y_true_all.append(row["burglaries"])
                y_pred_all.append(pred)

            df_results_list.extend(preds)

    df_results = pd.DataFrame(df_results_list)
    df_results = df_results.sort_values(by=["ward_name", "month_num"]).reset_index(drop=True)

    X_test = test_df[features] if 'test_df' in locals() else None
    return df_results, model, X_test
