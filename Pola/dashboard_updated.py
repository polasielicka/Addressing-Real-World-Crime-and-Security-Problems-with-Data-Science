import geopandas as gpd
import json
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
shapefile_path = os.path.join(script_dir, "..", "data", "coordinate_mapping_2025", "IMD_mapping_result.shp")
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

# Format months to display just Year-Month
forecast_df['month'] = pd.to_datetime(forecast_df['month'])
forecast_df['month_label'] = forecast_df['month'].dt.strftime('%Y-%m')

# Normalize ward names in GeoDataFrame for matching
gdf['NAME_clean'] = (
    gdf['NAME']
    .str.lower()
    .str.replace("&", "and")
    .str.replace(" ward", "")
    .str.strip()
)

# Merge forecast once globally
forecast_agg = forecast_df.groupby(['ward', 'month_label'])['forecast'].sum().reset_index()

# Precompute base GeoJSON
geojson_data = json.loads(gdf.to_json())

# Prepare dropdown options for month filtering
available_months = forecast_agg['month_label'].unique()
available_months.sort()

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("London Ward Boundaries and Forecasted Burglaries"),
    dcc.Dropdown(
        id="month-selector",
        options=[{"label": m, "value": m} for m in available_months],
        value=list(available_months),
        multi=True,
        placeholder="Select months to filter",
    ),
    html.Div(id="summary-output", style={"fontSize": "20px", "marginTop": "10px", "fontWeight": "bold"}),
    dcc.Graph(id="ward-map"),
    html.Div(id="click-output"),
    dcc.Tabs(id="tab-selector", value='ward', children=[
        dcc.Tab(label='Selected Ward Forecast', value='ward'),
        dcc.Tab(label='Top 10 Wards (Selected Months)', value='top10'),
        dcc.Tab(label='Monthly Totals', value='monthly')
    ]),
    dcc.Graph(id="forecast-bar", style={"marginTop": "30px"})
])

@app.callback(
    [Output("ward-map", "figure"),
     Output("click-output", "children"),
     Output("forecast-bar", "figure"),
     Output("summary-output", "children")],
    [Input("ward-map", "clickData"),
     Input("month-selector", "value"),
     Input("tab-selector", "value")],
    [State("ward-map", "figure")]
)
def update_dashboard(clickData, selected_months, selected_tab, current_map):
    filtered_df = forecast_agg[forecast_agg['month_label'].isin(selected_months)] if selected_months else forecast_agg.copy()

    # Map coloring data
    map_data = filtered_df.groupby('ward')['forecast'].sum().reset_index().rename(columns={'forecast': 'total_forecast'})
    gdf_copy = gdf.copy()
    gdf_copy = gdf_copy.merge(map_data, how='left', left_on='NAME_clean', right_on='ward')

    fig_map = px.choropleth_mapbox(
        gdf_copy,
        geojson=geojson_data,
        locations=gdf_copy.index,
        color="total_forecast",
        color_continuous_scale="Viridis",
        mapbox_style="carto-positron",
        zoom=9,
        center={"lat": 51.5074, "lon": -0.1278},
        opacity=0.6,
        hover_name="NAME",
        hover_data=["HECTARES", "IMDRank", "IMDDecil", "Index of M"]
    )
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    fig_bar = px.bar(title="Select a tab to display relevant chart")
    text = "Click on a ward to see its forecast."

    if selected_tab == 'ward' and clickData:
        idx = clickData["points"][0]["location"]
        ward = gdf.iloc[idx]
        ward_name_clean = ward['NAME_clean']
        text = f"Clicked Ward: {ward['NAME']}"
        ward_forecast = filtered_df[filtered_df['ward'] == ward_name_clean]
        if not ward_forecast.empty:
            fig_bar = px.bar(
                ward_forecast,
                x="month_label",
                y="forecast",
                title=f"Forecast for {ward['NAME']}",
                labels={"month_label": "Month", "forecast": "Predicted Burglaries"}
            )

    elif selected_tab == 'top10':
        top10 = (filtered_df.groupby('ward')['forecast'].sum()
                 .sort_values(ascending=False).head(10).reset_index())
        fig_bar = px.bar(
            top10,
            x="ward",
            y="forecast",
            title="Top 10 Wards by Predicted Burglaries",
            labels={"ward": "Ward", "forecast": "Total Forecast"}
        )

    elif selected_tab == 'monthly':
        monthly_totals = filtered_df.groupby('month_label')['forecast'].sum().reset_index()
        fig_bar = px.bar(
            monthly_totals,
            x="month_label",
            y="forecast",
            title="Monthly Total Burglaries",
            labels={"month_label": "Month", "forecast": "Total Forecast"}
        )

    total_forecast = filtered_df['forecast'].sum()
    summary_text = f"Total Predicted Burglaries in Selected Months: {int(total_forecast)}"

    return fig_map, text, fig_bar, summary_text

if __name__ == "__main__":
    app.run(debug=True)
