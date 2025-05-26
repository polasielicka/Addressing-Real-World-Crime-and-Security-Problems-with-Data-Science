import os
import glob
import pandas as pd
import geopandas as gpd
from Ian.add_ward import coordinate_mapping
from Ian.add_ward import lsoa_mapping

df_coordinate = coordinate_mapping()
df_lsoa = lsoa_mapping()

def get_unique_ward_names(df):
    return df["ward_name"].dropna().unique().tolist()

unique_wards_coordinate = get_unique_ward_names(df_coordinate) # list of all ward names that have a burglary
# print(len(unique_wards_coordinate))
# unique_wards_lsoa = get_unique_ward_names(df_lsoa)
# print(len(unique_wards_lsoa))


script_dir = os.path.dirname(os.path.abspath(__file__))
shapefile_path = os.path.join(script_dir, "..", "data_CBL", "coordinate_mapping_2018", "London_Ward.shp")
gdf = gpd.read_file(shapefile_path)
ward_names = sorted(gdf['NAME'].unique()) # 638 total wards
# print(len(ward_names))

burglary_counts = df_coordinate['ward_name'].value_counts()
# print(burglary_counts)
burglary_summary = burglary_counts.describe()
print(burglary_summary)

extra = list(set(ward_names) - set(unique_wards_coordinate)) # Lime street is the only ward with 0 burglaries
# print(extra[0])

wards_under_100 = burglary_counts[burglary_counts < 100].index.tolist() # 45 wards
print(wards_under_100)
print(len(wards_under_100))


city_of_london_wards = ["Aldersgate", "Aldgate", "Bassishaw", "Billingsgate", "Bishopsgate", "Bread Street",
                        "Bridge", "Broad Street", "Candlewick", "Castle Baynard", "Cheap", "Coleman Street",
                        "Cordwainer", "Cornhill", "Cripplegate", "Dowgate", "Farringdon Within", "Farringdon Without",
                        "Langbourn", "Lime Street", "Portsoken", "Queenhithe", "Tower", "Vintry", "Walbrook"]


df_city = df_coordinate[df_coordinate['ward_name'].isin(city_of_london_wards)]
ward_counts_filtered = df_city['ward_name'].value_counts()
ward_counts_filtered["Lime Street"] = 0
summary_stats = ward_counts_filtered.describe()
print("city of london ward counts:\n", ward_counts_filtered)
print("\nSummary statistics of city ward counts:\n", summary_stats)
print('')
df_metro = df_coordinate[~df_coordinate['ward_name'].isin(city_of_london_wards)]
ward_counts_filtered = df_metro['ward_name'].value_counts()
summary_stats = ward_counts_filtered.describe()
print("metropolitan area ward counts:\n", ward_counts_filtered)
print("\nSummary statistics of metro ward counts:\n", summary_stats)
