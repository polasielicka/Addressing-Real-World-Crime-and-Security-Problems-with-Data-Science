import pandas as pd
import os
import matplotlib.pyplot as plt

data_path = "data_CBL"
all_data = []

for folder in os.listdir(data_path):
    month_folder = os.path.join(data_path, folder)

    if os.path.isdir(month_folder):
        for file in os.listdir(month_folder):
            if "street" in file and file.endswith(".csv"):
                full_path = os.path.join(month_folder, file)
                df = pd.read_csv(full_path)
                df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
                all_data.append(df)

crime_df = pd.concat(all_data, ignore_index=True)
burglary_df = crime_df[crime_df['Crime type'] == 'Burglary']
monthly_counts = burglary_df.groupby(burglary_df["Month"].dt.to_period("M")).size()

monthly_counts.plot(kind='bar', figsize=(10, 6), title="Monthly Burglary Incidents")
plt.xlabel("Month")
plt.ylabel("Number of Burglaries")
plt.tight_layout()
plt.show()

outcome_data = []

for folder in os.listdir(data_path):
    month_folder = os.path.join(data_path, folder)

    if os.path.isdir(month_folder):
        for file in os.listdir(month_folder):
            if "outcomes" in file and file.endswith(".csv"):
                full_path = os.path.join(month_folder, file)
                df = pd.read_csv(full_path)
                outcome_data.append(df)

outcomes_df = pd.concat(outcome_data, ignore_index=True)
merged_df = pd.merge(burglary_df, outcomes_df, how='left', on=['Crime ID'])

outcome_counts = merged_df['Outcome type'].value_counts()

outcome_counts.plot(kind='bar', figsize=(10, 6), title="Burglary Outcome Types")
plt.xlabel("Outcome Type")
plt.ylabel("Number of Cases")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

