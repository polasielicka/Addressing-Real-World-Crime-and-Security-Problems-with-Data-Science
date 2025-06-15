import geopandas as gpd
import json
import plotly.express as px
import dash
from dash import dcc, html, Input, Output
import pandas as pd
import os
import calendar

script_dir = os.path.dirname(os.path.abspath(__file__))
shapefile_path = os.path.join(script_dir, "..", "data", "coordinate_mapping_2025", "IMD_mapping_result.shp")
gdf = gpd.read_file(shapefile_path)
gdf = gdf.to_crs(epsg=4326)

# Load XGBoost predictions
xgb_path = os.path.join(script_dir, "..", "output", "results.csv")
xgb_df = pd.read_csv(xgb_path)
xgb_df['ward'] = (
    xgb_df['ward_name']
    .str.lower()
    .str.replace("&", "and")
    .str.replace(" ward", "")
    .str.strip()
)
xgb_df['month_label'] = xgb_df['month_num'].apply(lambda m: calendar.month_name[m])

# Load SARIMA forecast data for overall prediction (next 12 months)
sarima_forecast_path = os.path.join(script_dir, "..", "output", "sarima_forecast.csv")
sarima_df = pd.read_csv(sarima_forecast_path)
sarima_df['month'] = pd.to_datetime(sarima_df['month'])
sarima_df['month_label'] = sarima_df['month'].dt.strftime('%B')

# Normalize names in GeoDataFrame
gdf['NAME_clean'] = (
    gdf['NAME']
    .str.lower()
    .str.replace("&", "and")
    .str.replace(" ward", "")
    .str.strip()
)

geojson_data = json.loads(gdf.to_json())
available_months = sorted(xgb_df['month_label'].unique(), key=lambda m: list(calendar.month_name).index(m))

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H2("London Ward Boundaries and Forecasted Burglaries"),

    html.Label("Select Months:", style={"fontWeight": "bold", "marginBottom": "5px"}),
    dcc.RangeSlider(
        id="month-selector",
        min=0,
        max=len(available_months) - 1,
        value=[0, len(available_months) - 1],
        marks={i: m for i, m in enumerate(available_months)},
        step=None
    ),

    dcc.Graph(id="ward-map"),

    dcc.Tabs(id="tab-selector", value='ward', children=[
        dcc.Tab(label='Selected Ward Prediction', value='ward'),
        dcc.Tab(label='Top 10 Wards', value='top10'),
        dcc.Tab(label='Overall Forecast for Special Operations', value='sarima')
    ]),

    html.Div(id="click-output", style={"marginTop": "10px", "fontWeight": "bold"}),
    dcc.Graph(id="forecast-bar", style={"marginTop": "30px"})
])

@app.callback(
    [Output("ward-map", "figure"),
     Output("click-output", "children"),
     Output("forecast-bar", "figure")],
    [Input("ward-map", "clickData"),
     Input("tab-selector", "value"),
     Input("month-selector", "value")]
)
def update_dashboard(clickData, selected_tab, selected_month_range):
    gdf_copy = gdf.copy()
    start_idx, end_idx = selected_month_range
    selected_months = available_months[start_idx:end_idx + 1]
    filtered_df = xgb_df[xgb_df['month_label'].isin(selected_months)]
    map_data = filtered_df.groupby('ward')['predicted_burglaries'].sum().reset_index().rename(columns={'predicted_burglaries': 'total_forecast'})
    gdf_copy = gdf_copy.merge(map_data, how='left', left_on='NAME_clean', right_on='ward')

    text = ""
    fig_bar = px.bar(title="Select a tab to display relevant chart")

    if selected_tab == 'ward' and clickData:
        idx = clickData["points"][0]["location"]
        ward = gdf.iloc[idx]
        ward_name = ward['NAME']
        ward_name_clean = ward['NAME_clean']
        pred_row = filtered_df[filtered_df['ward'] == ward_name_clean]
        if not pred_row.empty:
            monthly_sum = pred_row.groupby('month_label')['predicted_burglaries'].sum().reset_index()
            fig_bar = px.bar(
                monthly_sum,
                x="month_label",
                y="predicted_burglaries",
                title=f"Monthly Predictions for {ward_name}",
                labels={"month_label": "Month", "predicted_burglaries": "Predicted Burglaries"}
            )
            text = f"Clicked Ward: {ward_name}"

    elif selected_tab == 'top10':
        top10 = map_data.sort_values(by='total_forecast', ascending=False).head(10)
        fig_bar = px.bar(
            top10,
            x="ward",
            y="total_forecast",
            title="Top 10 Wards",
            labels={"ward": "Ward", "total_forecast": "Predicted Burglaries"}
        )

    elif selected_tab == 'sarima':
        fig_bar = px.line(
            sarima_df,
            x="month_label",
            y="forecast",
            title="City-wide Forecast for the Next 12 Months",
            labels={"month_label": "Month", "forecast": "Predicted Burglaries"}
        )

    if selected_tab != 'ward':
        text = ""

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
        hover_data={"HECTARES": True, "IMDRank": True, "IMDDecil": True, "Index of M": True, "total_forecast": False}
    )
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    return fig_map, text, fig_bar

if __name__ == "__main__":
    app.run(debug=True)
