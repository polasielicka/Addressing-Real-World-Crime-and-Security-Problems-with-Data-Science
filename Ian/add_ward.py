import os
import glob
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


def coordinate_mapping():
    """
    loads data and maps based on coordinates.
    152 out of 160k instances of burglary cannot be mapped because they fall outside of the map.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data', 'data_all', '*03')
    shapefile_path = os.path.join(script_dir, '..', "data", 'coordinate_mapping_2025', 'london_only_wards_2025.shp')
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


df = coordinate_mapping()
print(df.head())
print(len(df))
print(f"ward name is empty", len(df[df['ward_name'].isnull()]))
# print(f"number of different wards with burgalaries", df['ward_name'].nunique())
#
#
# # Filter the rows where 'ward_name' is null
# missing_ward_df = df[df['ward_name'].isnull()]
#
# # 1. Basic info and count
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


def dont_use_this(): # the old lsoa mapping
    """
    loads data and maps bases on external datasets that have mapped lsoas to wards.
    :return:
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    data_dir = os.path.join(script_dir, '..', 'data', 'crime_data')
    lookup_path_2024 = os.path.join(script_dir, '..', 'data', 'best_fit_lsoa_data (dont use)',
                                    'LSOA_(2021)_to_Electoral_Ward_(2024)_to_LAD_(2024)_Best_Fit_Lookup_in_EW.csv')
    lookup_path_2018 = os.path.join(script_dir, '..', 'data', 'best_fit_lsoa_data (dont use)',
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
    df['ward_name'] = df['ward2024'].fillna(df['ward2018'])

    # Filter for burglaries
    burglary_df = df[df["Crime type"].str.lower() == "burglary"]

    burglary_df = burglary_df[burglary_df['LSOA code'].notna()]

    return burglary_df
