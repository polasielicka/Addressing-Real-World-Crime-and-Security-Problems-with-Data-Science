import os
import glob
import pandas as pd

combined_data = glob.glob(os.path.join("../data_CBL", "*", "*-street.csv"))
df_lsoa = pd.concat((pd.read_csv(f) for f in combined_data), ignore_index=True)

script_dir = os.path.dirname(os.path.abspath(__file__))
shapefile_path = os.path.join(script_dir, '..', "Ian", "LSOA_(2021)_to_Electoral_Ward_(2024)_to_LAD_(2024)_Best_Fit_Lookup_in_EW.csv")
lsoa_lookup = pd.read_csv(shapefile_path)

lsoa_lookup.head()

df = df_lsoa.merge(
    lsoa_lookup[['LSOA21CD', 'WD24NM']],
    how='left',
    left_on='LSOA code',
    right_on='LSOA21CD'
)

df.rename(columns={'WD24NM': 'ward'}, inplace=True)

df.drop(columns=['LSOA21CD'], inplace=True)

burglary_df = df[df["Crime type"].str.lower() == "burglary"]

print(burglary_df.head())
print(burglary_df.columns)

missing_wards = burglary_df[burglary_df['ward'].isnull()]
print("Rows with missing ward values:")
print(missing_wards)
print(len(missing_wards))

missing_ward_lsoas = burglary_df[burglary_df['ward'].isnull()]['LSOA code'].unique()
print(missing_ward_lsoas)
print(len(missing_ward_lsoas))

