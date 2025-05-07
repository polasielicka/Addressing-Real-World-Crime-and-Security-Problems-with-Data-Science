import geopandas as gpd
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
shapefile_path = os.path.join(script_dir, '..', "Ian", 'London-wards-2018_ESRI', 'London_Ward.shp')
gdf = gpd.read_file(shapefile_path)

gdf.plot()
plt.title("London Wards")
plt.show()
