import os
import glob
import pandas as pd

combined_data = glob.glob(os.path.join("../data_CBL", "*", "*-street.csv"))
df_lsoa = pd.concat((pd.read_csv(f) for f in combined_data), ignore_index=True)

script_dir = os.path.dirname(os.path.abspath(__file__))
shapefile_path = os.path.join(script_dir, '..', "Ian", "LSOA_(2021)_to_Electoral_Ward_(2024)_to_LAD_(2024)_Best_Fit_Lookup_in_EW.csv")
lsoa_lookup = pd.read_csv(shapefile_path)

shapefile_path2 = os.path.join(script_dir, '..', "Ian", "Lower_Layer_Super_Output_Area_(2011)_to_Ward_(2018)_Lookup_in_England_and_Wales_v3.csv")
lsoa_lookup2 = pd.read_csv(shapefile_path2)

print(lsoa_lookup.head())

print(lsoa_lookup2.head())

df1 = df_lsoa.merge(
    lsoa_lookup[['LSOA21CD', 'WD24NM']],
    how='left',
    left_on='LSOA code',
    right_on='LSOA21CD'
)
df1.rename(columns={'WD24NM': 'ward2024'}, inplace=True)
df1.drop(columns=['LSOA21CD'], inplace=True)

df = df1.merge(
    lsoa_lookup2[['LSOA11CD', 'WD18NM']],
    how='left',
    left_on='LSOA code',
    right_on='LSOA11CD'
)
df.rename(columns={'WD18NM': 'ward2018'}, inplace=True)
df.drop(columns=['LSOA11CD'], inplace=True)

print(df.columns)
df['ward'] = df['ward2024'].fillna(df['ward2018'])

burglary_df = df[df["Crime type"].str.lower() == "burglary"]

print(burglary_df.head())
print(burglary_df.columns)

missing_ward_lsoas = burglary_df[burglary_df['ward'].isnull()]['Crime ID']
print(missing_ward_lsoas)

# row = df[df['Crime ID'] == '70e8e6ddbf8ee26c8e38997a45450b6bebf12f634ab799']
# print(row)
#
#
