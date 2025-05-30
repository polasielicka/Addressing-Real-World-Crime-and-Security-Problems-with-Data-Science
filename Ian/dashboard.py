import geopandas as gpd
import pandas as pd
import json
import plotly.express as px
import dash
from dash import dcc, html, Input, Output
import os

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', None)

script_dir = os.path.dirname(os.path.abspath(__file__))
shapefile_path = os.path.join(script_dir, "..", "data", "coordinate_mapping_2025", "london_only_wards_2025.shp")
gdf = gpd.read_file(shapefile_path)
gdf = gdf.to_crs(epsg=4326)
# print(len(london_wards))
# print(london_wards.head())


geojson_data = json.loads(gdf.to_json())

fig = px.choropleth_mapbox(
    gdf,
    geojson=geojson_data,
    locations=gdf.index,
    color_discrete_sequence=["#1f77b4"],
    mapbox_style="carto-positron",
    zoom=9,
    center={"lat": 51.5074, "lon": -0.1278},
    opacity=0.3,
    hover_name="NAME",
    hover_data=["HECTARES"]
)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})


app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("London Ward Boundaries"),
    dcc.Graph(id="ward-map", figure=fig),
    html.Div(id="click-output")
])

@app.callback(
    Output("click-output", "children"),
    Input("ward-map", "clickData")
)
def display_click_data(clickData):
    if clickData:
        idx = clickData["points"][0]["location"]
        ward = gdf.iloc[idx]
        return f"Clicked Ward: {ward['NAME']}"
    return "Placeholder, click on ward to show its name"

if __name__ == "__main__":
    app.run(debug=True)

