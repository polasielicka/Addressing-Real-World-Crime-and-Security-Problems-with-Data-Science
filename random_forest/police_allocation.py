import pandas as pd
import geopandas as gpd
import os
from pathlib import Path

def allocation():

    # Load predicted burglaries per ward/month
    df = pd.read_csv('output/results.csv')

    # get directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
    output_dir = os.path.abspath(os.path.join(script_dir, '..', 'output'))
    datasets_dir = os.path.join(project_root, 'CBL-group-5','datasets')

    ward_forecast = df.groupby('ward_name')['predicted_burglaries'].mean().rename('forecast')

    # Pivot to wide for easier recent calculation
    pivot = df.pivot(index='ward_name', columns='month_num', values='predicted_burglaries')
    max_month = pivot.columns.max()
    last = pivot[max_month]
    rolling_avg = pivot[max_month-3:max_month].mean(axis=1)
    recent_spike = ((last - rolling_avg) / rolling_avg).replace([pd.NA, pd.NaT], 0).rename('recent_spike')

    datasets_dir_path = Path(datasets_dir)
    csv_path = datasets_dir_path / "coordinate_mapping_2025" / "IMD_mapping_result.shp"
    if not csv_path.exists():
        raise FileNotFoundError(f"IMD CSV file not found at {csv_path}")
    df_imd = gpd.read_file(csv_path)

    # rename columns for clarity
    df_final = df_imd.rename(columns={"NAME": "ward_name"})

    # rename columns (shapefile column names are bugged)
    df_final = df_final.rename(columns={
        "ward_name": "ward_name",
        "IMDRank": "imd_rank",
        "IMDDecil": "imd_decile",
        "Index of M" : "imd_score",
        "Income Sco": "income_score",
        "Employment": "employment_domain_score",
        "Education,": "education_domain_score",
        "Health Dep": "health_domain_score",
        "Crime Scor": "crime_domain_score",
        "Barriers t": "housing_domain_score",
        "Living Env": "environment_domain_score"
    })

    # 4. Combine into risk composite
    # Make a DataFrame of your two Series
    metrics = (
        ward_forecast.rename("forecast").to_frame()
        .join(recent_spike.rename("recent_spike"))
        .reset_index()           # turn the ward_name index into a column
    )

    # Drop the geometry from df_final, then merge in the metrics
    data = (
        pd.merge(
        df_final.drop(columns="geometry"),
        metrics,
        on="ward_name",
        how="left"
        )
        .fillna(0)     # now only fills missing numeric/IMD values
    )

    data['risk_score'] = 0.7*data['forecast'] + 0.2*data['recent_spike'] + 0.1*data['imd_score']

    # 5. Assign risk tiers: top 20% High, next 60% Medium, bottom 20% Low
    n = len(data)
    data = data.sort_values('risk_score', ascending=False)
    data['tier'] = pd.qcut(data['risk_score'], q=[0, .2, .8, 1.0], labels=['High','Medium','Low'])

    # 6. Officer-hour allocation proportional to predicted burglaries
    tier_ranges     = {'High':(17,18), 'Medium':(6,7), 'Low':(1,2)}
    fmin, fmax      = data['forecast'].min(), data['forecast'].max()
    data['officers_per_shift'] = data.apply(
        lambda row: assign_officers(row['tier'], row['forecast'], tier_ranges, (fmin,fmax)),
        axis=1
    )

    # 8. Save — include ward_name so you can see who got what
    data[['ward_name',
          'forecast','officers_per_shift','tier'
         ]].to_csv('output/allocation.csv', index=False)

def assign_officers(tier, forecast, tier_ranges, forecast_bounds):
    """
    tier_ranges: {'High': (17,18), 'Medium': (6,7), 'Low': (1,2)}
    forecast_bounds: (min_forecast, max_forecast)
    """
    lo, hi = tier_ranges[tier]
    fmin, fmax = forecast_bounds
    # clamp forecast into [0,1]
    pct = (forecast - fmin) / (fmax - fmin) if fmax > fmin else 0
    # linearly map pct to [lo, hi]
    return int(round(lo + pct * (hi - lo)))

def main():
    allocation()

if __name__ == "__main__":
    main()