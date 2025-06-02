import os
import glob
import pandas as pd
import geopandas as gpd
from Ian.add_ward import coordinate_mapping

df_coordinate = coordinate_mapping()

def get_unique_ward_names(df):
    return df["ward_name"].dropna().unique().tolist()

unique_wards_coordinate = get_unique_ward_names(df_coordinate) # list of all ward names that have a burglary
# print(len(unique_wards_coordinate))
# unique_wards_lsoa = get_unique_ward_names(df_lsoa)
# print(len(unique_wards_lsoa))


script_dir = os.path.dirname(os.path.abspath(__file__))
shapefile_path = os.path.join(script_dir, "..", "data", "coordinate_mapping_2025", "london_only_wards_2025.shp")
gdf = gpd.read_file(shapefile_path)
ward_names = sorted(gdf['FILE_NAME'].unique()) # 692 out of 704 have burglaries
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
city_of_london_wards = [ward + " Ward" for ward in city_of_london_wards]


df_city = df_coordinate[df_coordinate['ward_name'].isin(city_of_london_wards)]
ward_counts_filtered = df_city['ward_name'].value_counts()
ward_counts_filtered["Lime Street"] = 0
summary_stats = ward_counts_filtered.describe()
print("city of london burglary counts:\n", ward_counts_filtered)
print("\nSummary statistics of city burglary counts:\n", summary_stats)
print('')
df_metro = df_coordinate[~df_coordinate['ward_name'].isin(city_of_london_wards)]
ward_counts_filtered = df_metro['ward_name'].value_counts()
summary_stats = ward_counts_filtered.describe()
print("metropolitan area burglary counts:\n", ward_counts_filtered)
print("\nSummary statistics of metro burglary counts:\n", summary_stats)
