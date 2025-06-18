import pandas as pd
from xgboost import XGBRegressor

# recursive forecasting imports
from recursive_forecasting import *


def hybrid_forecast(data, forecast_year, lag_range):
    # copy the data to avoid modifying the original
    df = data.copy()

    # run recursive forecasting to get predictions for the forecast year
    df_recursive, model_recursive, thing = recursive_forecast(df, forecast_years=[forecast_year], lag_range=lag_range)
    df_recursive = df_recursive.rename(columns={"predicted_burglaries": "recursive_pred"})
    df_recursive = df_recursive.drop(columns=["actual_burglaries"])

    # fill year column with forecast year
    df_recursive["year"] = forecast_year

    # Identify IMD features (static per ward)
    imd_features = [
        "imd_score",
        "income_score",
        "employment_domain_score",
        "education_domain_score",
        "health_domain_score",
        "crime_domain_score",
        "housing_domain_score",
        "environment_domain_score"
    ]
    imd_df = df[['ward_name'] + imd_features].drop_duplicates(subset='ward_name')

    # Split into training and testing by year (2023 and earlier for training, 2024 for testing)
    train_df = df[df["year"] < forecast_year]
    test_df = df[df["year"] == forecast_year]

    # Drop rows in train_df where any lag in lag_range or lag12 is NaN
    lag_cols = [f'lag{lag}' for lag in lag_range] + ['lag12']
    train_df = train_df.dropna(subset=lag_cols)

    # Aggregate both + add covid flag
    train_agg = train_df.groupby(["ward_name", "year", "month_num"], as_index=False)["burglaries"].mean()
    test_agg = test_df.groupby(["ward_name", "year", "month_num"], as_index=False)["burglaries"].mean()

    train_agg["covid_flag"] = ((train_agg["year"] > 2020) |
                               ((train_agg["year"] == 2020) & (train_agg["month_num"] >= 3))).astype(int)
    test_agg["covid_flag"] = ((test_agg["year"] > 2020) |
                              ((test_agg["year"] == 2020) & (test_agg["month_num"] >= 3))).astype(int)

    # Merge in IMD features
    train_agg = train_agg.merge(imd_df, on="ward_name", how="left")
    test_agg = test_agg.merge(imd_df, on="ward_name", how="left")

    # add real lag features to training set
    for lag in lag_range:
        train_agg[f'lag{lag}'] = (
            train_agg
            .groupby('ward_name')['burglaries']
            .shift(lag)
        )
    train_agg = train_agg.dropna(subset=[f'lag{lag}' for lag in lag_range])
    # add lag12 to training set
    train_agg['lag12'] = (
        train_agg
        .groupby('ward_name')['burglaries']
        .shift(12)
    )

    # Add historic data from train_agg 2023 to df_recursive
    historic_2023 = train_agg[["ward_name", "year", "month_num", "burglaries"]].copy()
    historic_2023 = historic_2023[historic_2023["year"] == 2023]
    historic_2023 = historic_2023.rename(columns={"burglaries": "recursive_pred"})
    df_recursive = pd.concat([historic_2023, df_recursive], ignore_index=True)

    # Add recursive predictions to test set
    for lag in lag_range:
        df_recursive[f'lag{lag}'] = (
            df_recursive
            .groupby('ward_name')['recursive_pred']
            .shift(lag)
        )
    df_recursive['lag12'] = (
        df_recursive
        .groupby('ward_name')['recursive_pred']
        .shift(12)
    )
    test_agg = test_agg.merge(
        df_recursive[["ward_name", "year", "month_num"] + [f"lag{lag}" for lag in lag_range] + ["lag12"]],
        on=["ward_name", "year", "month_num"],
        how="left"
    )
    test_agg = test_agg.dropna(subset=[f"lag{lag}" for lag in lag_range])

    # print columns in train_agg and test_agg
    print(f"Train columns: {train_agg.columns.tolist()}")
    print(f"Test columns: {test_agg.columns.tolist()}")

    # Prepare features and target
    feature_cols = imd_features + ["month_num", "covid_flag"] + [f"lag{lag}" for lag in lag_range] + ["lag12"]
    X_train = train_agg[feature_cols]
    y_train = train_agg["burglaries"]
    X_test = test_agg[feature_cols]
    y_test = test_agg["burglaries"]

    # Keep identifiers for results
    test_info = test_agg[["ward_name", "month_num"]].copy()

    # Train model (XGBoost with hyperparameters found w/ RandomizedSearchCV)
    model = XGBRegressor(
        n_estimators=450,
        learning_rate=0.03185412750221407,
        max_depth=6,
        subsample=0.792574766162736,
        colsample_bytree=0.5669260594003166,
        gamma=0.005468785930798914,
        min_child_weight=8.22255986735738,
        reg_alpha=0.7499107494699718,
        reg_lambda=1.8263315045152857,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Predict on 2023–2024
    y_pred = model.predict(X_test)

    # Prepare final results DataFrame
    results_df = test_info.copy()
    results_df["predicted_burglaries"] = y_pred
    results_df["actual_burglaries"] = y_test.values
    results_df = results_df.sort_values(by=["ward_name", "month_num"]).reset_index(drop=True)

    return results_df, model, X_test