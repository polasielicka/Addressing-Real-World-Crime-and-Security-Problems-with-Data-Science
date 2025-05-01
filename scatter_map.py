import os
import glob
import pandas   as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

street_glob  = os.path.join("data_CBL", "*", "*-street.csv")
outcome_glob = os.path.join("data_CBL", "*", "*-outcomes.csv")

df_street  = pd.concat((pd.read_csv(fp) for fp in glob.glob(street_glob)),  ignore_index=True)
df_outcome = pd.concat((pd.read_csv(fp) for fp in glob.glob(outcome_glob)), ignore_index=True)
df = pd.merge(df_street, df_outcome, on="Crime ID", how="inner")

lon_col = next((c for c in df.columns if "long" in c.lower()), None)
lat_col = next((c for c in df.columns if "lat"  in c.lower()), None)
if not lon_col or not lat_col:
    raise KeyError(f"Lon/Lat columns not found in {df.columns.tolist()}")

crime_col = next((c for c in df.columns if "crime type" in c.lower()), "Crime type")
df = (
    df
    [ df[crime_col].astype(str).str.strip().str.lower() == "burglary" ]
    .dropna(subset=[lon_col, lat_col])
    .copy()
)

gdf_burglary = gpd.GeoDataFrame(
    df,
    geometry = gpd.points_from_xy(df[lon_col], df[lat_col]),
    crs      = "EPSG:4326"
)

wards_url = (
    "https://martinjc.github.io/UK-GeoJSON/"
    "json/eng/wards_by_lad/topo_E09000001.json"
)
gdf_wards = gpd.read_file(wards_url)

if gdf_wards.crs is None:
    gdf_wards.set_crs(epsg=4326, inplace=True)

gdf_burglary = gdf_burglary.to_crs(epsg=3857)
gdf_wards    = gdf_wards.to_crs(epsg=3857)

fig, ax = plt.subplots(figsize=(12,12))

gdf_wards.boundary.plot(ax=ax, lw=1, edgecolor="gray", label="Ward boundary")

gdf_burglary.plot(
    ax=ax,
    markersize=6,
    alpha=0.6,
    color="red",
    label="Burglary"
)

ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, zoom=12)

ax.set_title("City of London Burglaries by Ward", fontsize=16)
ax.set_axis_off()
ax.legend()

plt.tight_layout()
plt.show()
