import os
import pandas as pd
import geopandas as gpd


# This calculates the average IMD rank and score for the wards in london based on the area. Doesnt need to be run anymore.

script_dir = os.path.dirname(os.path.abspath(__file__))
lsoa_shapefile = os.path.join(script_dir, '..', 'data', 'IMD', 'Lower_Super_Output_Area_(LSOA)_IMD2019_(WGS84).shp')
lsoa = gpd.read_file(lsoa_shapefile)

script_dir = os.path.dirname(os.path.abspath(__file__))
ward_shapefile = os.path.join(script_dir, '..', 'data', 'coordinate_mapping_2025', "london_only_wards_2025.shp")
wards = gpd.read_file(ward_shapefile)

pd.set_option('display.max_rows', None)      # Show all rows
pd.set_option('display.max_columns', None)   # Show all columns
pd.set_option('display.width', None)         # Don't limit width
pd.set_option('display.max_colwidth', None)  # Don't truncate column contents
print(lsoa.head())
print(lsoa.columns)
print(wards.head())
print(wards.columns)

projected_crs = "EPSG:27700"
lsoa = lsoa.to_crs(projected_crs)
wards = wards.to_crs(projected_crs)

intersection = gpd.overlay(wards, lsoa, how='intersection')

intersection['intersect_area'] = intersection.geometry.area
ward_areas = wards[['NAME', 'geometry']].copy()
ward_areas['ward_area'] = ward_areas.geometry.area

intersection = intersection.merge(ward_areas[['NAME', 'ward_area']], on='NAME')

intersection['weight'] = intersection['intersect_area'] / intersection['ward_area']

intersection['weighted_IMDRank'] = intersection['IMDRank'] * intersection['weight']
intersection['weighted_IMDDecil'] = intersection['IMDDecil'] * intersection['weight']

ward_imd = intersection.groupby('NAME')[['weighted_IMDRank', 'weighted_IMDDecil']].sum().reset_index()
wards_with_imd = wards.merge(ward_imd, on='NAME')

print(wards_with_imd.head())
print(wards_with_imd.columns)


# wards_with_imd.to_file("london_wards_area_weighted_IMD_CORRECTED.shp") # saves the new shapefile
