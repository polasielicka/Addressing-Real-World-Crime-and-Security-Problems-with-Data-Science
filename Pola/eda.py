import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import calendar
from statsmodels.graphics.tsaplots import plot_acf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(BASE_DIR, os.pardir, "data_CBL"))

def load_data(data=DATA_DIR):
    street_files = glob.glob(os.path.join(data, "*", "*-street.csv"))
    df = pd.concat((pd.read_csv(f) for f in street_files), ignore_index=True)
    month_col = next(c for c in df.columns if "month" in c.lower())
    df[month_col] = pd.to_datetime(df[month_col], format="%Y-%m")
    df = df[df["Crime type"].str.strip().str.lower() == "burglary"].copy()
    return df, month_col

def compute_monthly_counts(df, month_col):
    return df.set_index(month_col).resample("M").size()

def plot_monthly_bar(monthly_counts, title="Monthly Burglaries"):
    fig, ax = plt.subplots(figsize=(12,5))
    monthly_counts.plot(kind="bar", ax=ax)
    ax.set_xticklabels([dt.strftime("%Y-%m") for dt in monthly_counts.index], rotation=45, ha="right")
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Burglaries")
    fig.tight_layout()
    plt.show()

def plot_monthly_by_year_bar(df, month_col, title="Monthly Burglaries by Year"):
    df["Year"] = df[month_col].dt.year
    df["MonthNum"] = df[month_col].dt.month
    monthly_year = df.groupby(['MonthNum','Year']).size().unstack('Year', fill_value=0)
    monthly_year.index = [calendar.month_name[m] for m in monthly_year.index]
    fig, ax = plt.subplots(figsize=(12,6))
    monthly_year.plot(kind="bar", width=0.8, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Burglaries")
    ax.legend(title="Year")
    fig.tight_layout()
    plt.show()

def plot_monthly_by_year_stacked(df, month_col, title="Monthly Burglaries by Year (Stacked)"):
    df["Year"] = df[month_col].dt.year
    df["MonthNum"] = df[month_col].dt.month
    monthly_year = df.groupby(['MonthNum','Year']).size().unstack('Year', fill_value=0)
    monthly_year.index = [calendar.month_name[m] for m in monthly_year.index]
    fig, ax = plt.subplots(figsize=(12,6))
    monthly_year.plot(kind="bar", stacked=True, width=0.8, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Burglaries")
    ax.legend(title="Year", bbox_to_anchor=(1.02,1), loc="upper left")
    fig.tight_layout()
    plt.show()

def plot_time_series(monthly_counts, title="Monthly Burglary Time Series"):
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(monthly_counts.index, monthly_counts.values, marker="o")
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Burglaries")
    fig.autofmt_xdate()
    fig.tight_layout()
    plt.show()

def plot_avg_by_month(monthly_counts, title="Average Burglaries by Calendar Month"):
    counts = monthly_counts[monthly_counts.index.year != pd.Timestamp.now().year]
    avg_by_month = counts.groupby(counts.index.month).mean()
    months = range(1,13)
    fig, ax = plt.subplots(figsize=(12,5))
    ax.bar(months, avg_by_month.values)
    ax.set_xticks(months)
    ax.set_xticklabels([calendar.month_abbr[m] for m in months])
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Mean count")
    fig.tight_layout()
    plt.show()

def plot_boxplot_monthly(monthly_counts, title="Distribution of Monthly Counts by Month"):
    counts = monthly_counts[monthly_counts.index.year != pd.Timestamp.now().year]
    dfc = counts.to_frame(name="count")
    dfc['month'] = dfc.index.month
    fig, ax = plt.subplots(figsize=(12,5))
    dfc.boxplot(column='count', by='month', ax=ax)
    ax.set_xlabel("Month")
    ax.set_ylabel("Burglaries")
    ax.set_title(title)
    plt.suptitle("")
    fig.tight_layout()
    plt.show()

def plot_acf_monthly(monthly_counts, lags=24, title="Autocorrelation of Monthly Burglary Counts"):
    fig, ax = plt.subplots(figsize=(12,5))
    plot_acf(monthly_counts, lags=lags, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    plt.show()

def plot_year_on_year_overlay(monthly_counts, title="Year‐on‐Year Monthly Burglary Patterns"):
    dfc = monthly_counts.to_frame(name="count")
    dfc["year"] = dfc.index.year
    dfc["month"] = dfc.index.month
    fig, ax = plt.subplots(figsize=(12,6))
    for year, grp in dfc.groupby("year"):
        ax.plot(grp["month"], grp["count"], marker="o", label=str(year))
    ax.set_xticks(range(1,13))
    ax.set_xticklabels([calendar.month_abbr[m] for m in range(1,13)])
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Count")
    ax.legend(title="Year")
    fig.tight_layout()
    plt.show()

def main():
    df_burglary, month_col = load_data()
    monthly_counts = compute_monthly_counts(df_burglary, month_col)
    plot_monthly_bar(monthly_counts)
    plot_monthly_by_year_bar(df_burglary, month_col)
    plot_monthly_by_year_stacked(df_burglary, month_col)
    plot_time_series(monthly_counts)
    plot_avg_by_month(monthly_counts)
    plot_boxplot_monthly(monthly_counts)
    plot_acf_monthly(monthly_counts)
    plot_year_on_year_overlay(monthly_counts)

if __name__ == "__main__":
    main()
