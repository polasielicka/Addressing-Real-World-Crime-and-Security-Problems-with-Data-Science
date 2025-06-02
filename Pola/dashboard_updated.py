import geopandas as gpd
import json
import plotly.express as px
import dash
from dash import dcc, html, Input, Output
import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
shapefile_path = os.path.join(script_dir, "..", "data", "coordinate_mapping_2025", "london_wards_area_weighted_IMD_CORRECTED.shp")
gdf = gpd.read_file(shapefile_path)
gdf = gdf.to_crs(epsg=4326)

# Load forecast data
forecast_path = os.path.join(script_dir, "..", "Pola", "results", "ward_future_forecasts.csv")
forecast_df = pd.read_csv(forecast_path)
forecast_df['ward'] = (
    forecast_df['ward']
    .str.lower()
    .str.replace("&", "and")
    .str.replace(" ward", "")
    .str.strip()
)

# Also normalize ward names in GeoDataFrame for matching
gdf['NAME_clean'] = (
    gdf['NAME']
    .str.lower()
    .str.replace("&", "and")
    .str.replace(" ward", "")
    .str.strip()
)

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
    hover_data=["HECTARES", "weighted_I", "weighted_1"]
)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("London Ward Boundaries"),
    dcc.Graph(id="ward-map", figure=fig),
    html.Div(id="click-output"),
    dcc.Graph(id="forecast-bar", style={"marginTop": "30px"})
])

@app.callback(
    [Output("click-output", "children"),
     Output("forecast-bar", "figure")],
    [Input("ward-map", "clickData")]
)
def display_click_data(clickData):
    if clickData:
        idx = clickData["points"][0]["location"]
        ward = gdf.iloc[idx]
        ward_name_clean = ward['NAME_clean']

        text = f"Clicked Ward: {ward['NAME']}"

        forecast = forecast_df[forecast_df['ward'] == ward_name_clean]
        if not forecast.empty:
            fig_bar = px.bar(
                forecast,
                x="month",
                y="forecast",
                title=f"12-Month Forecast for {ward['NAME']}",
                labels={"month": "Month", "forecast": "Predicted Burglaries"}
            )
        else:
            fig_bar = px.bar(title="No forecast available")

        return text, fig_bar

    return "Placeholder, click on ward to show its name", px.bar(title="")

if __name__ == "__main__":
    app.run(debug=True)
