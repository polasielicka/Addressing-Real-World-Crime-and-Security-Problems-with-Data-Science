def load_data():
    import os
    import pandas as pd
    from datetime import datetime

    # root directory containing folders with CSV files
    root_dir = '../data_CBL/crime_data'  

    # list to hold dataframes
    df_list = []

    # loop through each folder in the root directory
    # and read all CSV files into dataframes, adding them to df_list
    for folder in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith('.csv'):
                    file_path = os.path.join(folder_path, file)
                    try:
                        df = pd.read_csv(file_path)
                        df['YearMonth'] = folder  # e.g. '2022-06'
                        df_list.append(df)
                    except Exception as e:
                        print(f"Error reading {file_path}: {e}")
    
    # concatenate all dataframes into one
    all_data = pd.concat(df_list, ignore_index=True)

    # select burglaries only
    burglary_data = all_data[all_data['Crime type'].str.lower() == 'burglary']

    # drop data beyond 2024
    burglary_data = burglary_data[burglary_data['YearMonth'].dt.year <= 2024]

    # proper datetime format
    burglary_data["Month"] = pd.to_datetime(burglary_data["Month"])   # 2024-09 -> 2024-09-01

    # one row per LSOA per month
    burg_counts = (
    burglary_data.groupby(["LSOA code", "Month"])
        .size()
        .reset_index(name="burglaries")
        .rename(columns={"LSOA code": "LSOA11CD"})
    )

    imd = pd.read_excel(
        "File_5_-_IoD2019_Scores.xlsx",
        sheet_name="IoD2019 Scores",
        usecols="A,E:T",               #  ← Excel-style range
        engine="openpyxl"
    ).rename(columns={"LSOA code (2011)": "LSOA11CD"})

    # monthly_counts = burglary_data.groupby(['YearMonth']).size().reset_index(name='BurglaryCount')
    return burg_counts

def main():
    data = load_data()
    print(data)

if __name__ == "__main__":
    main()