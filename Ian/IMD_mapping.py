import os
import pandas as pd
import geopandas as gpd

def imd_mapping():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    lsoa_shapefile = os.path.join(script_dir, '..', 'data', 'IMD', 'Lower_Super_Output_Area_(LSOA)_IMD2019_(WGS84).shp')
    lsoa = gpd.read_file(lsoa_shapefile)

    imd_path = os.path.join(script_dir, '..', 'data', 'IMD', "imd_scores_csv.csv")
    imd_df = pd.read_csv(imd_path)

    ward_shapefile = os.path.join(script_dir, '..', 'data', 'coordinate_mapping_2025', "london_only_wards_2025.shp")
    wards = gpd.read_file(ward_shapefile)

    # Identify duplicates
    duplicated_names = wards["NAME"][wards["NAME"].duplicated(keep=False)]
    wards["NAME"] = wards.apply(
        lambda row: f"{row['NAME']} ({row['CODE']})" if row["NAME"] in duplicated_names.values else row["NAME"],
        axis=1
    )

    projected_crs = "EPSG:27700"
    lsoa = lsoa.to_crs(projected_crs)
    wards = wards.to_crs(projected_crs)

    lsoa = lsoa.merge(imd_df, left_on='lsoa11cd', right_on='LSOA code (2011)', how='left')

    intersection = gpd.overlay(wards, lsoa, how='intersection')

    intersection['intersect_area'] = intersection.geometry.area
    ward_areas = wards[['NAME', 'geometry']].copy()
    ward_areas['ward_area'] = ward_areas.geometry.area

    intersection = intersection.merge(ward_areas[['NAME', 'ward_area']], on='NAME')

    intersection['weight'] = intersection['intersect_area'] / intersection['ward_area']

    intersection['IMDRank'] = intersection['IMDRank'] * intersection['weight']
    intersection['IMDDecil'] = intersection['IMDDecil'] * intersection['weight']
    intersection['Index of Multiple Deprivation (IMD) Score'] = intersection[
                                                                    'Index of Multiple Deprivation (IMD) Score'] * \
                                                                intersection['weight']
    intersection['Income score (rate)'] = intersection['Income Score (rate)'] * intersection['weight']
    intersection['Employment score (rate)'] = intersection['Employment Score (rate)'] * intersection['weight']
    intersection['Education, Skills and Training Score'] = intersection['Education, Skills and Training Score'] * \
                                                           intersection['weight']
    intersection['Health Deprivation and Disability Score'] = intersection['Health Deprivation and Disability Score'] * \
                                                              intersection['weight']
    intersection['Crime Score'] = intersection['Crime Score'] * intersection['weight']
    intersection['Barriers to Housing and Services Score'] = intersection['Barriers to Housing and Services Score'] * \
                                                             intersection['weight']
    intersection['Living Environment Score'] = intersection['Living Environment Score'] * intersection['weight']

    ward_imd = intersection.groupby('NAME')[['IMDRank', 'IMDDecil', 'Index of Multiple Deprivation (IMD) Score',
                                             'Income Score (rate)', 'Employment Score (rate)',
                                             'Education, Skills and Training Score',
                                             'Health Deprivation and Disability Score', 'Crime Score',
                                             'Barriers to Housing and Services Score',
                                             'Living Environment Score']].sum().reset_index()
    wards_with_imd = wards.merge(ward_imd, on='NAME')

    return wards_with_imd


wards_with_imd = imd_mapping()
# Save to CSV
OUTPUT_PATH = "../data/coordinate_mapping_2025/IMD_mapping_result.shp"
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
wards_with_imd.to_file(OUTPUT_PATH, index=False)
print(f"Forecast saved to: {OUTPUT_PATH}")
