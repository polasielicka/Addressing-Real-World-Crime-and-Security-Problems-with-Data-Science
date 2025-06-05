import os
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def load_data():

    # root directory containing folders with CSV files
    root_dir = 'data/crime_data'  

    # list to hold dataframes
    df_list = []

    # loop through each folder in the root directory
    # and read all CSV files into dataframes, adding them to df_list
    for folder in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.csv'):
                    file_path = os.path.join(folder_path, file)
                    try:
                        df = pd.read_csv(file_path)
                        df['YearMonth'] = folder  # e.g. '2022-06'
                        df_list.append(df)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
    
    # concatenate all dataframes into one
    all_data = pd.concat(df_list, ignore_index=True)

    # select burglaries only
    burglary_data = all_data[all_data['Crime type'].str.lower() == 'burglary'].copy()

    # proper datetime format
    burglary_data["Month"] = pd.to_datetime(burglary_data["YearMonth"])

    # drop data beyond 2024
    burglary_data = burglary_data[burglary_data['Month'].dt.year <= 2024]

    # one row per LSOA per month
    burg_counts = (
    burglary_data.groupby(["LSOA code", "Month"])
        .size()
        .reset_index(name="burglaries")
        .rename(columns={"LSOA code": "LSOA11CD"})
    )

    # read IMD data
    imd_path = "data/IMD/imd_scores.xlsx" 
    imd = pd.read_excel(
        imd_path,
        sheet_name="IoD2019 Scores",
        usecols="A,E:T",
        engine="openpyxl"
    ).rename(columns={"LSOA code (2011)": "LSOA11CD"})

    # merge IMD data with burglary counts
    merged = pd.merge(burg_counts, imd, how='left', on='LSOA11CD')

    # report missing IMD data
    report_missing_imd(merged)

    merged_clean = pd.merge(burg_counts, imd, how='left', on='LSOA11CD').query("`Index of Multiple Deprivation (IMD) Score`.notna()").copy()

    lookup_path = "data/best_fit_lsoa_data (dont use)/Lower_Layer_Super_Output_Area_(2011)_to_Ward_(2018)_Lookup_in_England_and_Wales_v3.csv"
    lsoa2ward = (
        pd.read_csv(lookup_path, usecols=["LSOA11CD","WD18NM"])
          .rename(columns={"WD18NM": "ward"})
    )

    # merge LSOA to ward mapping
    df = merged_clean.merge(lsoa2ward, on="LSOA11CD", how="left")
    # report missing ward mappings
    missing = df['ward'].isna().sum()
    print(f"Rows without a ward mapping: {missing}")

    # drop any rows that failed to map
    df = df[df['ward'].notna()]

    # aggregate by ward and month
    imd_feats = [c for c in df.columns
                 if c not in ('LSOA11CD','Month','burglaries','ward')]
    agg_dict = {c: 'mean' for c in imd_feats}
    agg_dict['burglaries'] = 'sum'

    # group by ward and month, aggregating the IMD features and burglaries
    ward_df = (
        df
        .groupby(['ward','Month'], as_index=False)
        .agg(agg_dict)
    )

    return ward_df

def report_missing_imd(merged_df):
    missing_imd = merged_df[merged_df["Index of Multiple Deprivation (IMD) Score"].isna()]
    num_missing_lsoas = missing_imd["LSOA11CD"].nunique()
    total_lsoas = merged_df["LSOA11CD"].nunique()
    print(f"Number of LSOAs without IMD data: {num_missing_lsoas} out of {total_lsoas} total LSOAs.")

def prepare_train_test_split(df):
    df = df.copy()
    
    # extract year and month_num
    df["year"] = df["Month"].dt.year
    df["month_num"] = df["Month"].dt.month

    # drop month column
    df = df.drop(columns=["Month"])

    # IMD features
    imd_features = [col for col in df.columns if 'Score' in col or 'Domain' in col or 'IMD' in col]

    # aggregate by ward, year, and month_num (averaging IMD, summing burglaries)
    agg_dict = {col: 'mean' for col in imd_features}
    agg_dict["burglaries"] = "sum"

    grouped_df = (
        df
        .groupby(["ward", "year", "month_num"], as_index=False)
        .agg(agg_dict)
    )

    return grouped_df



def train_random_forest(data):
    df = data.copy()

    # Identify IMD features (static per ward)
    imd_features = [col for col in df.columns if 'Score' in col or 'Domain' in col or 'IMD' in col]
    imd_df = df[['ward'] + imd_features].drop_duplicates(subset='ward')
    
    # Split into training and testing by year (2023 and earlier for training, 2024 for testing)
    train_df = df[df["year"] <= 2023]
    test_df = df[df["year"] >= 2024]

    # Aggregate both + add covid flag
    train_agg = train_df.groupby(["ward", "year", "month_num"], as_index=False)["burglaries"].mean()
    train_agg["covid_flag"] = (train_agg["year"] >= 2020).astype(int)
    test_agg = test_df.groupby(["ward", "year", "month_num"], as_index=False)["burglaries"].mean()
    test_agg["covid_flag"] = (test_agg["year"] >= 2020).astype(int)

    # Merge IMD into both
    train_agg = train_agg.merge(imd_df, on="ward", how="left")
    test_agg = test_agg.merge(imd_df, on="ward", how="left")

    # Prepare features and target
    feature_cols = imd_features + ["month_num", "covid_flag"]
    X_train = train_agg[feature_cols]
    y_train = train_agg["burglaries"]
    X_test = test_agg[feature_cols]
    y_test = test_agg["burglaries"]

    # alternatively, do a stanard train-test split
    # X = df[feature_cols]
    # y = df["burglaries"]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Keep identifiers for results
    test_info = test_agg[["ward", "month_num"]].copy()

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict on 2023–2024
    y_pred = model.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"Evaluation on 2023–2024:")
    print(f"  Mean Squared Error (MSE): {mse:.2f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"  R-squared (R²): {r2:.2f}")

    # Feature importances
    importances = model.feature_importances_
    sorted_idx = importances.argsort()
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), importances[sorted_idx])
    plt.yticks(range(len(sorted_idx)), [feature_cols[i] for i in sorted_idx])
    plt.xlabel("Feature Importance")
    plt.title("Random Forest Feature Importances")
    plt.tight_layout()
    plt.show()

    # Prepare final results DataFrame
    results_df = test_info.copy()
    results_df["predicted_burglaries"] = y_pred
    results_df["actual_burglaries"] = y_test.values
    results_df = results_df.sort_values(by=["ward", "month_num"]).reset_index(drop=True)

    print("Description of Test Data:")
    print(y_test.describe())

    return results_df, model

def main():
    # load the data
    print("Loading data...")
    data = load_data()
    data = prepare_train_test_split(data)
    print(f"Data loaded with {data.shape[0]} rows and {data.shape[1]} columns.")

    # save the data to an excel file
    os.makedirs("output", exist_ok=True) # avoid error
    data.to_excel("output/data.xlsx", index=False)
    print("Data saved to data.xlsx")

    # train random forest model
    print("Training Random Forest model...")
    df_results, model = train_random_forest(data)
    print(f"Results saved with {df_results.shape[0]} rows and {df_results.shape[1]} columns.")

    # save results to an excel file
    df_results.to_excel("output/results.xlsx", index=False)
    print("Results saved to results.xlsx")

    # model saved to model variable

if __name__ == "__main__":
    main()