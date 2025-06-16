import pandas as pd
import os
from pathlib import Path

def assign_officers(tier, forecast, tier_ranges, forecast_bounds):
    """
    tier_ranges: mapping of tier label to (min_officers, max_officers)
    forecast_bounds: (min_forecast, max_forecast) for current month
    """
    lo, hi = tier_ranges[tier]
    fmin, fmax = forecast_bounds

    pct = (forecast - fmin) / (fmax - fmin) if fmax > fmin else 0
    return int(round(lo + pct * (hi - lo)))


def allocation_per_month(input_csv: str, output_csv: str):
    
    # Load monthly predictions
    df = pd.read_csv(input_csv)

    # Define tiers and officer ranges
    tier_labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    tier_ranges = {
        'Very High': (17, 18),
        'High':      (9, 10),
        'Medium':    (6, 7),
        'Low':       (3, 4),
        'Very Low':  (1, 2)
    }

    results = []
    # Process month by month
    for month, group in df.groupby('month_num'):
        month_df = group[['ward_name', 'predicted_burglaries']].copy()
        month_df.rename(columns={'predicted_burglaries': 'forecast'}, inplace=True)

        # assign quintile tiers within this month
        month_df = month_df.sort_values('forecast', ascending=False)
        month_df['tier'] = pd.qcut(
            month_df['forecast'], 
            q=[0, .2, .4, .6, .8, 1.0],
            labels=tier_labels
        )

        # compute bounds for officer mapping
        fmin, fmax = month_df['forecast'].min(), month_df['forecast'].max()

        # assign officers per 2h shift
        month_df['officers_per_shift'] = month_df.apply(
            lambda row: assign_officers(
                row['tier'], row['forecast'], tier_ranges, (fmin, fmax)
            ), axis=1
        )

        month_df['month_num'] = month
        results.append(month_df)

    # concatenate and save
    out = pd.concat(results, ignore_index=True)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Saved monthly allocation to {output_csv}")


def main():
    input_csv = os.path.join('output', 'results.csv')
    output_csv = os.path.join('output', 'monthly_allocation.csv')
    allocation_per_month(input_csv, output_csv)

if __name__ == '__main__':
    main()
