import geopandas as gpd
import json
import plotly.express as px
import dash
from dash import dcc, html, Input, Output
import pandas as pd
import os

# Load spatial data
script_dir = os.path.dirname(os.path.abspath(__file__))
shapefile_path = os.path.join(script_dir, "..", "data", "coordinate_mapping_2018", "London_Ward.shp")
gdf = gpd.read_file(shapefile_path).to_crs(epsg=4326)
geojson_data = json.loads(gdf.to_json())

# Load forecast data
forecast_path = os.path.join(script_dir, "..", "Pola", "results", "ward_future_forecasts.csv")
metrics_path = os.path.join(script_dir, "..", "Pola", "results", "ward_forecast_metrics.csv")
future_df = pd.read_csv(forecast_path)
metrics_df = pd.read_csv(metrics_path)

# Standardize ward names
future_df['ward'] = future_df['ward'].str.lower().str.replace("&", "and").str.strip()
metrics_df['ward'] = metrics_df['ward'].str.lower().str.replace("&", "and").str.strip()

# Initialize map
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
    hover_data=["DISTRICT", "HECTARES"]
)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

# Start Dash app
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
def display_ward_forecast(clickData):
    if clickData:
        idx = clickData["points"][0]["location"]
        ward = gdf.iloc[idx]
        ward_name = ward['NAME'].lower().replace("&", "and").strip()
        district = ward['DISTRICT']

        # Forecast data
        forecast = future_df[future_df['ward'] == ward_name]

        # Metric data
        metrics = metrics_df[metrics_df['ward'] == ward_name]
        if not metrics.empty:
            row = metrics.iloc[0]
            smape = row['smape']
            mase = row['mase']
            reliability = "High" if smape < 30 and mase < 1 else ("Medium" if smape < 50 else "Low")
            text = f"Clicked Ward: {ward['NAME']} ({district})\nForecast Reliability: {reliability}\nSMAPE: {smape:.1f}%\nMASE: {mase:.2f}"
        else:
            text = f"Clicked Ward: {ward['NAME']} ({district})\nNo forecast metrics available."

        # Forecast bar chart
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

    return "Click on a ward to see its forecast.", px.bar(title="")

if __name__ == "__main__":
    app.run_server(debug=True)
