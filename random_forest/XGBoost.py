import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from xgboost import plot_importance
from pathlib import Path
from sklearn.model_selection import GridSearchCV

def load_data():
    # get directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    output_dir = os.path.abspath(os.path.join(script_dir, '..', 'output'))
    datasets_dir = os.path.join(project_root, 'CBL-group-5','data')

    # read input burglary data from csv
    all_data_path = os.path.join(output_dir, 'input_data.csv')
    if not os.path.exists(all_data_path):
        raise FileNotFoundError(f"'input_data.csv' not found at {all_data_path}. Please ensure the file exists.")
    all_data = pd.read_csv(all_data_path) # potentially clunky

    # initial cleaning step
    burglary_data = all_data[all_data['Crime type'].str.lower() == 'burglary'].copy() # select burglaries only
    burglary_data["Month"] = pd.to_datetime(burglary_data["Month"]) # proper datetime format
    burglary_data = burglary_data[burglary_data['Month'].dt.year <= 2024] # drop data beyond 2024

    # aggregate burglaries by ward and month
    burg_counts = (
        burglary_data.groupby(["ward_name", "Month"])
        .size()
        .reset_index(name="burglaries")
    )

    # read IMD data
    datasets_dir_path = Path(datasets_dir)
    csv_path = datasets_dir_path / "coordinate_mapping_2025" / "IMD_mapping_result.shp"
    if not csv_path.exists():
        raise FileNotFoundError(f"IMD CSV file not found at {csv_path}")
    df_imd = gpd.read_file(csv_path)

    # rename columns for clarity
    df_imd = df_imd.rename(columns={"NAME": "ward_name"})

    # merge LSOA to ward mapping
    df_merged = df_imd.merge(burg_counts, on="ward_name", how="left")

    # report and drop missing ward mappings
    missing = df_merged['ward_name'].isna().sum()
    df_final = df_merged[df_merged['ward_name'].notna()]
    print(f"Rows without a ward mapping: {missing}")

    # drop unnecessary columns
    cols_to_drop = [
        "AREA_CODE", "DESCRIPTIO", "FILE_NAME", "NUMBER", "NUMBER0", "POLYGON_ID",
        "UNIT_ID", "CODE", "HECTARES", "AREA", "TYPE_CODE", "DESCRIPT0",
        "TYPE_COD0", "DESCRIPT1", "geometry"
    ]
    df_final = df_final.drop(columns=[col for col in cols_to_drop if col in df_final.columns])

    # rename columns (shapefile column names are bugged)
    df_final = df_final.rename(columns={
        "ward_name": "ward_name",
        "IMDRank": "imd_rank",
        "IMDDecil": "imd_decile",
        "Index of M" : "imd_score",
        "Income Sco": "income_score",
        "Employment": "employment_domain_score",
        "Education,": "education_domain_score",
        "Health Dep": "health_domain_score",
        "Crime Scor": "crime_domain_score",
        "Barriers t": "housing_domain_score",
        "Living Env": "environment_domain_score"
    })

    return df_final

def prepare_train_test_split(df):
    df = df.copy()

    # extract year and month_num
    df["year"] = df["Month"].dt.year
    df["month_num"] = df["Month"].dt.month

    # drop month column
    df = df.drop(columns=["Month"])

    # IMD features
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

    # aggregate by ward, year, and month_num (averaging IMD, summing burglaries)
    agg_dict = {col: 'mean' for col in imd_features}
    agg_dict["burglaries"] = "sum"

    grouped_df = (
        df
        .groupby(["ward_name", "year", "month_num"], as_index=False)
        .agg(agg_dict)
    )

    print(grouped_df.head())

    # -----------------------------------
    # Missing months and wards fix
    # -----------------------------------
    df = grouped_df.reset_index()

    all_months = pd.date_range(start="2011-12-01", end="2024-12-01", freq="MS")
    all_months_df = pd.DataFrame({
        'year': all_months.year,
        'month_num': all_months.month
    })

    wards = df['ward_name'].unique()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    datasets_dir = os.path.join(project_root, 'CBL-group-5','data')
    datasets_dir_path = Path(datasets_dir)
    csv_path = datasets_dir_path / "coordinate_mapping_2025" / "IMD_mapping_result.shp"

    if not csv_path.exists():
        raise FileNotFoundError(f"IMD CSV file not found at {csv_path}")
    df_imd = gpd.read_file(csv_path)

    # rename columns for clarity
    df_imd = df_imd.rename(columns={"NAME": "ward_name"})
    duplicates = df_imd[df_imd.duplicated('ward_name', keep=False)]
    print(duplicates['ward_name'].unique())

    # new dataset
    full_data = []

    imd_cols = ['imd_score', 'income_score', 'employment_domain_score', 'education_domain_score',
                'health_domain_score', 'crime_domain_score', 'housing_domain_score', 'environment_domain_score']

    for ward in wards:
        ward_df = df[df['ward_name'] == ward]

        if ward_df.empty:
            continue

        # if month is missing, add month with buglaries = 0
        imd_values = {col: ward_df[col].iloc[0] if col in ward_df.columns else 0 for col in imd_cols}

        ward_months = all_months_df.copy()
        ward_months['ward_name'] = ward
        for col, val in imd_values.items():
            ward_months[col] = val

        merged = pd.merge(ward_months, ward_df, on=['ward_name', 'year', 'month_num'], how='left')

        merged['burglaries'] = merged['burglaries'].fillna(0).astype(int)

        for col in imd_cols:
            if f"{col}_x" in merged.columns and f"{col}_y" in merged.columns:
                merged[col] = merged[f"{col}_x"]
                merged.drop([f"{col}_x", f"{col}_y"], axis=1, inplace=True)

        merged = merged[['ward_name', 'year', 'month_num'] + imd_cols + ['burglaries']]
        full_data.append(merged)

    final_df = pd.concat(full_data)
    final_df = final_df.sort_values(by=['ward_name', 'year', 'month_num'])

    return final_df

def train_random_forest(data):
    df = data.copy()

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
    train_df = df[df["year"] <= 2023]
    test_df = df[df["year"] >= 2024]

    # Aggregate both + add covid flag
    train_agg = train_df.groupby(["ward_name", "year", "month_num"], as_index=False)["burglaries"].mean()
    train_agg["covid_flag"] = (train_agg["year"] >= 2020).astype(int)
    test_agg = test_df.groupby(["ward_name", "year", "month_num"], as_index=False)["burglaries"].mean()
    test_agg["covid_flag"] = (test_agg["year"] >= 2020).astype(int)

    # Merge IMD into both
    train_agg = train_agg.merge(imd_df, on="ward_name", how="left")
    test_agg = test_agg.merge(imd_df, on="ward_name", how="left")

    # Prepare features and target
    feature_cols = imd_features + ["month_num", "covid_flag"]
    X_train = train_agg[feature_cols]
    y_train = train_agg["burglaries"]
    X_test = test_agg[feature_cols]
    y_test = test_agg["burglaries"]

    # Keep identifiers for results
    test_info = test_agg[["ward_name", "month_num"]].copy()

    # TO UNCOMMENT !!!
    # Train model (XGBoost without GridSearch)
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

    # model = XGBRegressor(
    #     n_estimators=200,
    #     learning_rate=0.05,
    #     max_depth=6,
    #     subsample=0.8,
    #     colsample_bytree=0.8,
    #     random_state=42
    # )

    model.fit(X_train, y_train)

    # best_model, rs_obj = random_search_xgboost(X_train, y_train)
    # model = best_model


    # Train model (using GridsearchCV for hyperparameter tuning)
    # param_grid = {
    #     'n_estimators': [100, 200, 300],
    #     'learning_rate': [0.01, 0.05, 0.1],
    #     'max_depth': [4, 6, 8],
    #     'subsample': [0.7, 0.8, 1.0],
    #     'colsample_bytree': [0.7, 0.8, 1.0]
    # }

    # base_model = XGBRegressor(random_state=42)
    # grid_search = GridSearchCV(
    #     estimator=base_model,
    #     param_grid=param_grid,
    #     cv=3,
    #     scoring='neg_mean_squared_error',
    #     n_jobs=-1,
    #     verbose=1
    # )
    # grid_search.fit(X_train, y_train)
    # model = grid_search.best_estimator_
    # print("Best parameters found:", grid_search.best_params_)
    # model.fit(X_train, y_train)
    ################################################################

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
    plot_importance(model, max_num_features=20, importance_type='weight')
    plt.title("Feature Importances (Gain)")
    plt.tight_layout()
    plt.show()

    # Prepare final results DataFrame
    results_df = test_info.copy()
    results_df["predicted_burglaries"] = y_pred
    results_df["actual_burglaries"] = y_test.values
    results_df = results_df.sort_values(by=["ward_name", "month_num"]).reset_index(drop=True)

    return results_df, model

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

def random_search_xgboost(X_train, y_train,
                          n_iter: int = 150,
                          cv: int = 3,
                          random_state: int = 42):

   
    param_dist = {
        "n_estimators": randint(100, 600),          
        "learning_rate": uniform(0.01, 0.29),        
        "max_depth": randint(3, 10),               
        "min_child_weight": uniform(0.5, 9.5),     
        "subsample": uniform(0.5, 0.5),           
        "colsample_bytree": uniform(0.5, 0.5),     
        "gamma": uniform(0, 0.4),                  
        "reg_alpha": uniform(0, 1.0),          
        "reg_lambda": uniform(0, 2.0)              
    }

    base_model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",          # fast histogram algorithm (CPU)
        random_state=random_state,
        n_jobs=-1
    )

    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=n_iter,
        cv=cv,
        scoring="neg_mean_squared_error",
        verbose=1,
        n_jobs=-1,
        random_state=random_state
    )

    random_search.fit(X_train, y_train)

    print("\nBest hyper-parameters found by RandomizedSearchCV:")
    for k, v in random_search.best_params_.items():
        print(f"   • {k:<18}: {v}")

    best_model = random_search.best_estimator_

    return best_model, random_search


def main():
    # load the data
    print("Loading data...")
    data = load_data()
    data = prepare_train_test_split(data)
    print(f"Data loaded with {data.shape[0]} rows and {data.shape[1]} columns.")

    # save the cleaned data to a csv file
    os.makedirs("output", exist_ok=True) # avoid error
    data.to_csv("output/data_cleaned.csv", index=False)
    print("Data saved to data_cleaned.csv")

    # train random forest model
    print("Training XGBoost model...")
    df_results, model = train_random_forest(data)
    print(f"Results saved with {df_results.shape[0]} rows and {df_results.shape[1]} columns.")

    # save results to a csv file
    df_results.to_csv("output/results.csv", index=False)
    print("Results saved to results.csv")

    # model saved to model variable

if __name__ == "__main__":
    main()
