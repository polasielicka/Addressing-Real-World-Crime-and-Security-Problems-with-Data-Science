import os
import glob
import pandas as pd


def load_burglary_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    data_dir = os.path.join(script_dir, '..', 'data_CBL')
    lookup_path_2024 = os.path.join(script_dir, '..', 'Ian',
                                    'LSOA_(2021)_to_Electoral_Ward_(2024)_to_LAD_(2024)_Best_Fit_Lookup_in_EW.csv')
    lookup_path_2018 = os.path.join(script_dir, '..', 'Ian',
                                    'Lower_Layer_Super_Output_Area_(2011)_to_Ward_(2018)_Lookup_in_England_and_Wales_v3.csv')

    # Load crime data
    combined_data = glob.glob(os.path.join(data_dir, "*", "*-street.csv"))
    df_lsoa = pd.concat((pd.read_csv(f) for f in combined_data), ignore_index=True)

    # Load lookup tables
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

    # Combine ward information
    df['ward'] = df['ward2024'].fillna(df['ward2018'])

    # Filter for burglaries
    burglary_df = df[df["Crime type"].str.lower() == "burglary"]

    # removes rows that dont have location data
    initial_row_count = len(burglary_df)
    burglary_df = burglary_df[burglary_df['LSOA code'].notna()]
    removed_rows = initial_row_count - len(burglary_df)
    print(f"Removed {removed_rows} rows with missing 'LSOA code'.")

    return burglary_df


# every row that doesnt have missing location data has a ward entry
df = load_burglary_data()
missing_wards = df[df['ward'].isnull()]
print(len(missing_wards))

# there are 2977 rows that dont have location data
missing_counts = df.isna().sum()
print(missing_counts)

# the location value for these is "No Location"
unique_locations = df[df['LSOA code'].isna()]['Location'].dropna().unique()
print(unique_locations)

# 162358 total burglaries
print(len(df))