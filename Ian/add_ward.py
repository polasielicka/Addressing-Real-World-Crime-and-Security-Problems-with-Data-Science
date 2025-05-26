import os
import glob
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
#test
def lsoa_mapping():
    """
    loads data and maps bases on external datasets that have mapped lsoas to wards.
    :return:
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    data_dir = os.path.join(script_dir, '..', 'data_CBL', 'crime_data')
    lookup_path_2024 = os.path.join(script_dir, '..', 'data_CBL', 'best_fit_lsoa_data',
                                    'LSOA_(2021)_to_Electoral_Ward_(2024)_to_LAD_(2024)_Best_Fit_Lookup_in_EW.csv')
    lookup_path_2018 = os.path.join(script_dir, '..', 'data_CBL', 'best_fit_lsoa_data',
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

def coordinate_mapping():
    """
    loads data and maps based on coordinates.
    152 out of 160k instances of burglary cannot be mapped because they fall outside of the map.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data_CBL', 'crime_data')
    shapefile_path = os.path.join(script_dir, '..', "data_CBL", 'coordinate_mapping_2018', 'London_Ward.shp')

    # Load crime data
    combined_data = glob.glob(os.path.join(data_dir, "*", "*-street.csv"))
    df_lsoa = pd.concat((pd.read_csv(f) for f in combined_data), ignore_index=True)


    geometry = [Point(xy) for xy in zip(df_lsoa["Longitude"], df_lsoa["Latitude"])]
    gdf_points = gpd.GeoDataFrame(df_lsoa.copy(), geometry=geometry, crs='EPSG:4326')  # WGS84

    # Load ward shapefile
    wards = gpd.read_file(shapefile_path)

    # Ensure coordinate systems match
    if wards.crs != gdf_points.crs:
        wards = wards.to_crs(gdf_points.crs)

    # Spatial join to find the ward each point falls in
    joined = gpd.sjoin(gdf_points, wards[['NAME', 'geometry']], how='left', predicate='within')

    # Add the ward name as a new column
    df_with_wards = joined.drop(columns='geometry').rename(columns={'NAME': 'ward_name'})

    # Filter for burglaries
    df_with_wards = df_with_wards[df_with_wards["Crime type"].str.lower() == "burglary"]

    df_with_wards = df_with_wards[df_with_wards['Longitude'].notna()]

    return df_with_wards


# # every row that doesnt have missing location data has a ward entry
# df = load_burglary_data()
# missing_wards = df[df['ward'].isnull()]
# print(len(missing_wards))
#
# # there are 2977 rows that dont have location data
# missing_counts = df.isna().sum()
# print(missing_counts)
#
# # the location value for these is "No Location"
# unique_locations = df[df['LSOA code'].isna()]['Location'].dropna().unique()
# print(unique_locations)
#
# # 162358 total burglaries
# print(len(df))

df = coordinate_mapping()
print(df.head())
# print(len(df[df['ward_name'].isnull()]))
# print(df[df['ward_name'].isnull()])
# print(df['ward_name'].nunique())
# unique_locations = df[df['ward_name'].isna()]['index_right'].unique()
# print(unique_locations)
# print(len(unique_locations))
# print(df[df['ward_name'].isna()]['Longitude'].nunique())
# print(df.isna().sum())
# print(df[df['ward_name'].isnull()].describe())

# Filter the rows where 'ward_name' is null
# missing_ward_df = df[df['ward_name'].isnull()]

# 1. Basic info and count
# print("Number of rows with missing ward_name:", len(missing_ward_df))
# print("\nColumn-wise null value count in those rows:")
# print(missing_ward_df.isnull().sum())
#
# # 2. Summary statistics for numeric columns
# print("\nSummary statistics for numeric columns:")
# print(missing_ward_df.describe())
#
# # 3. Summary for non-numeric columns
# print("\nTop values for non-numeric columns:")
# for col in missing_ward_df.select_dtypes(include=['object', 'category']).columns:
#     print(f"\nValue counts for '{col}':")
#     print(missing_ward_df[col].value_counts(dropna=False).head(10))
#
# # 4. If geographic/location data exists (e.g., lat/lon), check for those
# if 'latitude' in missing_ward_df.columns and 'longitude' in missing_ward_df.columns:
#     print("\nGeographic summary:")
#     print(missing_ward_df[['latitude', 'longitude']].describe())
#
# # 5. Optional: Check a few example rows
# print("\nSample rows with missing ward_name:")
# print(missing_ward_df.head(5))


df2 = lsoa_mapping()
print(df2.head())
# # print(df2[df2['ward'].isnull()])
# # print(df2[df2['ward']].unique)
# print(df2.isna().sum())


# def count_column_differences(df: pd.DataFrame, df2: pd.DataFrame, col1: str = "ward_name", col2: str = "ward") -> int:
#
#     min_len = min(len(df), len(df2))
#
#     series1 = df[col1].iloc[:min_len].reset_index(drop=True)
#     series2 = df2[col2].iloc[:min_len].reset_index(drop=True)
#
#     differences = series1 != series2
#     return differences.sum()
#
# # print(count_column_differences(df, df2))
