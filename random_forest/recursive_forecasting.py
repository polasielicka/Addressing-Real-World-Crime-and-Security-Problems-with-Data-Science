import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from xgboost import plot_importance
from sklearn.metrics import mean_squared_error, r2_score
import os

def load_data():
    return pd.read_csv("output/data_cleaned.csv")

def add_lags(df, lags=[1, 2, 3]):
    df = df.sort_values(["ward_name", "year", "month_num"])
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("ward_name")["burglaries"].shift(lag)
    return df

def add_covid_flag(df):
    # exclude first 2 months of 2020
    df["covid_flag"] = ((df["year"] > 2020) | ((df["year"] == 2020) & (df["month_num"] > 2))).astype(int)
    return df

def recursive_forecast_pipeline(df, forecast_years=[2024, 2025]):
    df = add_lags(df)
    df = add_covid_flag(df)
    
    df_results_list = []
    model = None

    features = [
    "imd_score", "income_score", "employment_domain_score",
    "education_domain_score", "health_domain_score", "crime_domain_score",
    "housing_domain_score", "environment_domain_score",
    "lag_1", "lag_2", "lag_3", "covid_flag",
    "month_num"
    ]

    # We'll collect true and predicted values to evaluate later
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

        for month in range(1, 13):
            month_df = test_df[test_df["month_num"] == month].copy()
            preds = []

            for idx, row in month_df.iterrows():
                ward = row["ward_name"]
                past_data = df[(df["ward_name"] == ward) & (
                    (df["year"] < year) | ((df["year"] == year) & (df["month_num"] < month))
                )].sort_values(["year", "month_num"]).tail(3)

                if len(past_data) < 3:
                    # Skip if not enough past data to fill lags
                    continue

                lags = past_data["burglaries"].values[::-1]  # reverse for lag_1, lag_2, lag_3
                if len(lags) < 3:
                    continue

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

    # Evaluation
    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    mse = mean_squared_error(y_true_all, y_pred_all)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_all, y_pred_all)

    print(f"Evaluation on {forecast_years}:")
    print(f"  Mean Squared Error (MSE): {mse:.2f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"  R-squared (R²): {r2:.2f}")

    # Feature importances plot
    plot_importance(model, max_num_features=20, importance_type='weight')
    plt.title("Feature Importances (Weight)")
    plt.tight_layout()
    plt.show()

    df_results = pd.DataFrame(df_results_list)
    df_results = df_results.sort_values(by=["ward_name", "month_num"]).reset_index(drop=True)

    return df_results, model