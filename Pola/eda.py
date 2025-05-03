import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import calendar
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf

def load_data(data="data_CBL"):
    street_files = glob.glob(os.path.join(data, "*", "*-street.csv"))
    outcome_files = glob.glob(os.path.join(data, "*", "*-outcomes.csv"))
    df_street = pd.concat((pd.read_csv(f) for f in street_files), ignore_index=True)
    df_outcome = pd.concat((pd.read_csv(f) for f in outcome_files), ignore_index=True)
    df = pd.merge(df_street, df_outcome, on="Crime ID", how="inner")
    month_col = next((c for c in df.columns if "month" in c.lower()), None)
    df[month_col] = pd.to_datetime(df[month_col], format="%Y-%m")
    df_burglary = df[df["Crime type"].str.strip().str.lower() == "burglary"].copy()
    return df_burglary, month_col

def compute_monthly_counts(df_burglary, month_col):
    return df_burglary.set_index(month_col).resample("M").size()

def plot_monthly_bar(monthly_counts):
    fig, ax = plt.subplots(figsize=(12, 5))
    monthly_counts.plot(kind="bar", ax=ax)
    ax.set_xticklabels([dt.strftime("%Y-%m") for dt in monthly_counts.index], rotation=45, ha="right")
    ax.set_title("Monthly Burglary Incidents")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Burglaries")
    fig.tight_layout()
    plt.show()


def plot_outcome_types(df_burglary):
    outcome_counts = df_burglary["Last outcome category"].value_counts()
    fig, ax = plt.subplots(figsize=(12, 5))
    outcome_counts.plot(kind="bar", ax=ax)
    ax.set_xticklabels(outcome_counts.index, rotation=45, ha="right")
    ax.set_title("Burglary Outcome Types")
    ax.set_xlabel("Outcome category")
    ax.set_ylabel("Number of Cases")
    fig.tight_layout()
    plt.show()


def plot_monthly_by_year_bar(df_burglary, month_col):
    df_burglary["Year"] = df_burglary[month_col].dt.year
    df_burglary["MonthNum"] = df_burglary[month_col].dt.month
    monthly_year = (
        df_burglary
        .groupby(['MonthNum', 'Year'])
        .size()
        .unstack('Year', fill_value=0)
    )
    monthly_year.index = [calendar.month_name[m] for m in monthly_year.index]

    fig, ax = plt.subplots(figsize=(12, 6))
    monthly_year.plot(kind='bar', width=0.8, ax=ax)
    ax.set_title("Monthly Burglaries by Year")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Burglaries")
    ax.legend(title='Year')
    fig.tight_layout()
    plt.show()


def plot_monthly_by_year_stacked(df_burglary, month_col):
    df_burglary["Year"] = df_burglary[month_col].dt.year
    df_burglary["MonthNum"] = df_burglary[month_col].dt.month
    monthly_year = (
        df_burglary
        .groupby(['MonthNum', 'Year'])
        .size()
        .unstack('Year', fill_value=0)
    )
    monthly_year.index = [calendar.month_name[m] for m in monthly_year.index]

    fig, ax = plt.subplots(figsize=(12, 6))
    monthly_year.plot(kind='bar', stacked=True, width=0.8, ax=ax)
    ax.set_title("Monthly Burglaries by Year (Stacked)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Burglaries")
    ax.legend(title='Year', bbox_to_anchor=(1.02, 1), loc='upper left')
    fig.tight_layout()
    plt.show()

def plot_time_series(monthly_counts):
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(monthly_counts.index, monthly_counts.values, marker="o")
    ax.set_title("Monthly Burglary Time Series")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Burglaries")
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()


def plot_avg_by_month(monthly_counts):
    counts = monthly_counts[monthly_counts.index.year != 2025]
    avg_by_month = counts.groupby(counts.index.month).mean()
    months = range(1, 13)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(months, avg_by_month.values)
    ax.set_xticks(months)
    ax.set_xticklabels([calendar.month_abbr[m] for m in months])
    ax.set_title("Average Burglaries by Calendar Month (without 2025)")
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean count")
    fig.tight_layout()
    plt.show()


def plot_boxplot_monthly(monthly_counts):
    counts = monthly_counts[monthly_counts.index.year != 2025]
    df_counts = counts.to_frame(name="count")
    df_counts['month'] = df_counts.index.month

    fig, ax = plt.subplots(figsize=(12, 5))
    df_counts.boxplot(column='count', by='month', ax=ax)
    ax.set_xlabel("Month")
    ax.set_ylabel("Burglaries")
    ax.set_title("Distribution of Monthly Counts by Calendar Month (without 2025)")
    plt.suptitle("")
    fig.tight_layout()
    plt.show()


def plot_seasonal_decompose(monthly_counts):
    ts = monthly_counts.copy()
    ts.index = pd.DatetimeIndex(ts.index, freq="M")
    decomp = seasonal_decompose(ts, model='additive')
    fig = decomp.plot()
    fig.set_size_inches(12, 8)
    fig.tight_layout()
    plt.show()


def plot_acf_monthly(monthly_counts, lags=24):
    fig, ax = plt.subplots(figsize=(12, 5))
    plot_acf(monthly_counts, lags=lags, ax=ax)
    ax.set_title("Autocorrelation of Monthly Burglary Counts")
    fig.tight_layout()
    plt.show()


def plot_year_on_year_overlay(monthly_counts):
    df_counts = monthly_counts.to_frame(name="count")
    df_counts["year"] = df_counts.index.year
    df_counts["month"] = df_counts.index.month

    fig, ax = plt.subplots(figsize=(12, 6))
    for year, grp in df_counts.groupby("year"):
        ax.plot(grp["month"], grp["count"], marker="o", label=str(year))
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels([calendar.month_abbr[m] for m in range(1, 13)])
    ax.set_title("Year‐on‐year Monthly Burglary Patterns")
    ax.set_xlabel("Month")
    ax.set_ylabel("Count")
    ax.legend(title="Year")
    fig.tight_layout()
    plt.show()


def main():
    df_burglary, month_col = load_data()
    monthly_counts = compute_monthly_counts(df_burglary, month_col)
    plot_monthly_bar(monthly_counts)
    plot_outcome_types(df_burglary)
    plot_monthly_by_year_bar(df_burglary, month_col)
    plot_monthly_by_year_stacked(df_burglary, month_col)
    plot_time_series(monthly_counts)
    plot_avg_by_month(monthly_counts)
    plot_boxplot_monthly(monthly_counts)
    plot_seasonal_decompose(monthly_counts)
    plot_acf_monthly(monthly_counts)
    plot_year_on_year_overlay(monthly_counts)


if __name__ == "__main__":
    main()
