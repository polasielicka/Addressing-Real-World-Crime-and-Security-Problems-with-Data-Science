import pandas as pd
import os
import matplotlib.pyplot as plt

data_path = "data_CBL"
all_data = []

for folder in os.listdir(data_path):
    month_folder = os.path.join(data_path, folder)

    if os.path.isdir(month_folder):
        for file in os.listdir(month_folder):
            if "street" in file and file.endswith(".csv"):
                full_path = os.path.join(month_folder, file)
                df = pd.read_csv(full_path)
                df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
                all_data.append(df)

crime_df = pd.concat(all_data, ignore_index=True)
burglary_df = crime_df[crime_df['Crime type'] == 'Burglary']
monthly_counts = burglary_df.groupby(burglary_df["Month"].dt.to_period("M")).size()

monthly_counts.plot(kind='bar', figsize=(10, 6), title="Monthly Burglary Incidents")
plt.xlabel("Month")
plt.ylabel("Number of Burglaries")
plt.tight_layout()
plt.show()

# Filter for burglary only (you can be more specific if needed)
burglary_df = crime_df[crime_df['Crime type'] == 'Burglary']

# Read the outcome files and merge if needed
outcome_data = []

for folder in os.listdir(data_path):
    month_folder = os.path.join(data_path, folder)

    if os.path.isdir(month_folder):
        for file in os.listdir(month_folder):
            if "outcomes" in file and file.endswith(".csv"):
                full_path = os.path.join(month_folder, file)
                df = pd.read_csv(full_path)
                outcome_data.append(df)

outcomes_df = pd.concat(outcome_data, ignore_index=True)
merged_df = pd.merge(burglary_df, outcomes_df, how='left', on=['Crime ID'])

outcome_counts = merged_df['Outcome type'].value_counts()

outcome_counts.plot(kind='bar', figsize=(10, 6), title="Burglary Outcome Types")
plt.xlabel("Outcome Type")
plt.ylabel("Number of Cases")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import os

# --- STEP 1: Load burglary street-level data ---
data_path = "data_CBL"
street_data = []

for folder in os.listdir(data_path):
    month_folder = os.path.join(data_path, folder)
    if os.path.isdir(month_folder):
        for file in os.listdir(month_folder):
            if "street" in file and file.endswith(".csv"):
                df = pd.read_csv(os.path.join(month_folder, file))
                df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
                street_data.append(df)

street_df = pd.concat(street_data, ignore_index=True)
street_df = street_df[street_df["Crime type"] == "Burglary"]

# --- STEP 2: Load and merge outcomes data ---
outcome_data = []

for folder in os.listdir(data_path):
    month_folder = os.path.join(data_path, folder)
    if os.path.isdir(month_folder):
        for file in os.listdir(month_folder):
            if "outcomes" in file and file.endswith(".csv"):
                df = pd.read_csv(os.path.join(month_folder, file))
                outcome_data.append(df)

outcomes_df = pd.concat(outcome_data, ignore_index=True)

# Merge on 'Crime ID' (preserve lat/lon from street_df)
merged_df = pd.merge(street_df, outcomes_df, how='left', on='Crime ID')

# --- STEP 3: Drop rows with missing location ---
merged_df = merged_df.dropna(subset=["Latitude", "Longitude"])

# --- STEP 4: Convert to GeoDataFrame ---
gdf_burglary = gpd.GeoDataFrame(
    merged_df,
    geometry=gpd.points_from_xy(merged_df.Longitude, merged_df.Latitude),
    crs="EPSG:4326"
)

# --- STEP 5: Load London Wards GeoJSON ---
# You can download from: https://data.london.gov.uk/download/statistical-gis-boundary-files-london/ward_boundaries.geojson
wards = gpd.read_file("london_wards.geojson")  # Update path as needed

# --- STEP 6: Plot ---
fig, ax = plt.subplots(figsize=(12, 10))
wards.plot(ax=ax, color='white', edgecolor='black', linewidth=0.5)
gdf_burglary.plot(ax=ax, markersize=5, color='red', alpha=0.5, label="Burglary")

plt.title("Burglary Incidents in London with Ward Boundaries")
plt.legend()
plt.axis('off')
plt.tight_layout()
plt.show()
