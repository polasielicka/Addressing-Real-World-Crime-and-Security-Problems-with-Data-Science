import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

street_files  = glob.glob(os.path.join("data_CBL", "*", "*-street.csv"))
outcome_files = glob.glob(os.path.join("data_CBL", "*", "*-outcomes.csv"))

df_street  = pd.concat((pd.read_csv(f) for f in street_files),  ignore_index=True)
df_outcome = pd.concat((pd.read_csv(f) for f in outcome_files), ignore_index=True)

df = pd.merge(df_street, df_outcome, on="Crime ID", how="inner")

print("Available columns:", df.columns.tolist())

month_col = next((c for c in df.columns if "month" in c.lower()), None)
if month_col is None:
    raise KeyError(f"Couldn't find any column with 'month' in its name. Found: {df.columns.tolist()}")

print(f"Using '{month_col}' as the Month column")

df[month_col] = pd.to_datetime(df[month_col], format="%Y-%m")

df_burglary = df[df["Crime type"].str.strip().str.lower() == "burglary"].copy()

monthly_counts = (
    df_burglary
    .set_index(month_col)
    .resample("M")
    .size()
    .rename("Count")
)
monthly_counts.index = monthly_counts.index.strftime("%Y-%m")

plt.figure(figsize=(12, 5))
monthly_counts.plot(kind="bar")
plt.xticks(rotation=45, ha="right")
plt.title("Monthly Burglary Incidents")
plt.xlabel("Month")
plt.ylabel("Number of Burglaries")
plt.tight_layout()
plt.show()

outcome_counts = df_burglary["Last outcome category"].value_counts()

plt.figure(figsize=(12,5))
outcome_counts.plot(kind="bar")
plt.xticks(rotation=45, ha="right")
plt.title("Burglary Outcome Types")
plt.xlabel("Outcome category")
plt.ylabel("Number of Cases")
plt.tight_layout()
plt.show()

import calendar
df_burglary['Year']     = df_burglary[month_col].dt.year
df_burglary['MonthNum'] = df_burglary[month_col].dt.month

monthly_year = (
    df_burglary
      .groupby(['MonthNum', 'Year'])
      .size()
      .unstack('Year', fill_value=0)
)

monthly_year.index = monthly_year.index.map(lambda m: calendar.month_name[m])

plt.figure(figsize=(12,6))
monthly_year.plot(kind='bar', width=0.8, figsize=(12,6))
plt.title("Monthly Burglaries by Year")
plt.xlabel("Month")
plt.ylabel("Number of Burglaries")
plt.xticks(rotation=45, ha='right')
plt.legend(title='Year')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,6))

monthly_year.plot(
    kind='bar',
    stacked=True,
    width=0.8,
    figsize=(12,6)
)

plt.title("Monthly Burglaries by Year (Stacked)")
plt.xlabel("Month")
plt.ylabel("Number of Burglaries")
plt.xticks(rotation=45, ha='right')
plt.legend(title='Year', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.tight_layout()
plt.show()
