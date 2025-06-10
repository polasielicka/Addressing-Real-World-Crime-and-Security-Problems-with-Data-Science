import geopandas as gpd
import json
import plotly.express as px
import dash
from dash import dcc, html, Input, Output
import pandas as pd
import os

# ----------------------------
# Load shapefile and GeoJSON
# ----------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
shapefile_path = os.path.join(script_dir, "..", "data", "coordinate_mapping_2025", "IMD_mapping_result.shp")
gdf = gpd.read_file(shapefile_path)
gdf = gdf.to_crs(epsg=4326)

# Standardize ward names IMPORTANT!!!
gdf["ward"] = gdf["NAME"].copy().str.replace(r"\s+ward$", "", case=False, regex=True).str.strip().str.lower()

# sort on ward names in alphabetical order
gdf = gdf.sort_values(by="ward").reset_index(drop=True)

# save Gdf to excel for debugging purposes
gdf.to_excel(os.path.join(script_dir, "..", "output", "s.xlsx"), index=False)

# ----------------------------
# Load results
# ----------------------------
results_path = os.path.join(script_dir, "..", "output", "results.csv")
df_results = pd.read_csv(results_path)
df_results["ward"] = df_results["ward_name"].str.strip().str.lower()

# ----------------------------
# Setup Dash
# ----------------------------
app = dash.Dash(__name__)
app.title = "Burglary Heatmap"

app.layout = html.Div([
    html.H2("London Ward Burglary Predictions Heatmap"),

    html.Label("Select Month:"),
    dcc.Slider(
        id="month-slider",
        min=1,
        max=12,
        value=1,
        marks={i: str(i) for i in range(1, 13)},
        step=1
    ),

    html.Label("View:"),
    dcc.RadioItems(
        id="view-toggle",
        options=[
            {"label": "Predicted Burglaries", "value": "predicted_burglaries"},
            {"label": "Actual Burglaries", "value": "actual_burglaries"},
        ],
        value="predicted_burglaries",
        labelStyle={'display': 'inline-block', 'margin-right': '15px'}
    ),

    dcc.Graph(id="ward-map"),

    html.Div(id="click-output")
])

# ----------------------------
# Callback for interactive map
# ----------------------------
@app.callback(
    Output("ward-map", "figure"),
    Output("click-output", "children"),
    Input("month-slider", "value"),
    Input("view-toggle", "value"),
    Input("ward-map", "clickData")
)
def update_map(month, value_column, clickData):
    # Filter data for selected month
    month_data = df_results[df_results["month_num"] == month][["ward", value_column]]

    # Merge with GeoDataFrame
    merged = gdf.merge(month_data, on="ward", how="left")

    # Build map
    fig = px.choropleth_map(
        merged,
        geojson=json.loads(merged.to_json()),
        locations=merged.index,
        color=value_column,
        hover_name="NAME",
        color_continuous_scale="Reds",
        opacity=0.6,
        map_style="carto-positron",
        zoom=9,
        center={"lat": 51.5074, "lon": -0.1278},
        height=600
    )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

    # Handle click display
    if clickData:
        idx = clickData["points"][0]["location"]
        ward_name = gdf.iloc[idx]["NAME"]
        return fig, f"Clicked Ward: {ward_name}"
    else:
        return fig, "Click on a ward to see its name"

# ----------------------------
# Run server
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)