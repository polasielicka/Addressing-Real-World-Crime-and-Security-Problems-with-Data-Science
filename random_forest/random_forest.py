import os
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def load_data():

    # root directory containing folders with CSV files
    root_dir = 'CBL-group-5/data/crime_data'  

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
    imd_path = "CBL-group-5/data/IMD/imd_scores.xlsx" 
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

    lookup_path = "CBL-group-5/data/best_fit_lsoa_data/Lower_Layer_Super_Output_Area_(2011)_to_Ward_(2018)_Lookup_in_England_and_Wales_v3.csv"
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

def aggregate_to_ward_month(df):
    df = df.copy()
    
    df["Month"] = df["Month"].apply(lambda dt: dt.replace(year=2000, day=1))
    
    agg_dict = {
        "burglaries": "sum",
        "Index of Multiple Deprivation (IMD) Score": "mean",
        "Income Score (rate)": "mean",
        "Employment Score (rate)": "mean",
        "Education, Skills and Training Score": "mean",
        "Health Deprivation and Disability Score": "mean",
        "Crime Score": "mean",
        "Barriers to Housing and Services Score": "mean",
        "Living Environment Score": "mean",
        # add other IMD columns as needed
    }

    ward_month_df = (
        df
        .groupby(["ward", "Month"], as_index=False)
        .agg(agg_dict)
    )

    return ward_month_df



def train_random_forest(data):
    df = data.copy()

    # keep identifiers for merging back later
    identifiers = df[["ward", "Month"]].copy()

    # clean and prepare the data
    df['year'] = df['Month'].dt.year
    df['month_num'] = df['Month'].dt.month
    df = df.drop(columns=['Month'])

    # select features and target variable
    imd_features = [col for col in df.columns if 'Score' in col or 'rate' in col or 'Domain' in col]
    temporal_features = ['year', 'month_num']
    X = df[imd_features + temporal_features]
    y = df['burglaries']

    # add ward and month to X for later rejoining
    X = X.copy()
    X["ward"] = identifiers["ward"]
    X["Month"] = identifiers["Month"]


    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # keep ward/month separate
    test_info = X_test[["ward", "Month"]].copy()

    # drop non-feature columns before training
    X_train = X_train.drop(columns=["ward", "Month"])
    X_test = X_test.drop(columns=["ward", "Month"])

    # train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # make predictions
    y_pred = model.predict(X_test)

    # build a result DataFrame
    results_df = test_info.copy()
    results_df["predicted_burglaries"] = y_pred
    results_df["actual_burglaries"] = y_test.values

    # evaluate performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.4f}")

    # plot feature importances
    print("Plotting feature importances...")
    importances = model.feature_importances_
    feat_names = X_train.columns  # <-- use X_train columns here

    # visualize feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(feat_names, importances)
    plt.xlabel("Feature Importance")
    plt.title("Random Forest - Feature Importances")
    plt.tight_layout()
    plt.show()

    return results_df

def main():
    # load the data
    print("Loading data...")
    data = load_data()
    data = aggregate_to_ward_month(data)
    print(f"Data loaded with {data.shape[0]} rows and {data.shape[1]} columns.")

    # save the data to an excel file
    os.makedirs("CBL-group-5/output", exist_ok=True) # avoid error
    data.to_excel("CBL-group-5/output/data.xlsx", index=False)
    print("Data saved to data.xlsx")

    # train random forest model
    print("Training Random Forest model...")
    results = train_random_forest(data)
    print(f"Results saved with {results.shape[0]} rows and {results.shape[1]} columns.")

    # save results to an excel file
    results.to_excel("CBL-group-5/output/results.xlsx", index=False)
    print("Results saved to results.xlsx")

if __name__ == "__main__":
    main()
