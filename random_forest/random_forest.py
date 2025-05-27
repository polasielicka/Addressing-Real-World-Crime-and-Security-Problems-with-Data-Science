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
    imd_path = "CBL-group-5/data/IMD/imd_scores.xlsx"  # update to the actual Excel file name
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

    return merged_clean

def report_missing_imd(merged_df):
    missing_imd = merged_df[merged_df["Index of Multiple Deprivation (IMD) Score"].isna()]
    num_missing_lsoas = missing_imd["LSOA11CD"].nunique()
    total_lsoas = merged_df["LSOA11CD"].nunique()
    print(f"Number of LSOAs without IMD data: {num_missing_lsoas} out of {total_lsoas} total LSOAs.")

def train_random_forest(data):
    df = data.copy()

    # clean and prepare the data
    df['year'] = df['Month'].dt.year
    df['month_num'] = df['Month'].dt.month
    df = df.drop(columns=['Month'])

    # select features and target variable
    imd_features = [col for col in df.columns if 'Score' in col or 'rate' in col or 'Domain' in col]
    temporal_features = ['year', 'month_num']
    X = df[imd_features + temporal_features]
    y = df['burglaries']

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # train the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    # make predictions
    y_pred = model.predict(X_test)

    # evaluate performance
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.2f}")
    print(f"R² Score: {r2:.4f}")

    # plot feature importances
    print("Plotting feature importances...")
    importances = model.feature_importances_
    feat_names = X.columns

    # visualize feature importances
    plt.figure(figsize=(10, 6))
    plt.barh(feat_names, importances)
    plt.xlabel("Feature Importance")
    plt.title("Random Forest - Feature Importances")
    plt.tight_layout()
    plt.show()

def main():
    # load the data
    print("Loading data...")
    data = load_data()
    print(f"Data loaded with {data.shape[0]} rows and {data.shape[1]} columns.")

    # save the data to an excel file
    os.makedirs("CBL-group-5/output", exist_ok=True) # avoid error
    data.to_excel("CBL-group-5/output/data.xlsx", index=False)
    print("Data saved to data.xlsx")

    # train random forest model
    print("Training Random Forest model...")
    train_random_forest(data)

if __name__ == "__main__":
    main()