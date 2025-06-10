
import os
from Ian.add_ward import coordinate_mapping


def load_data():
    # get directory paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(script_dir, '..'))

    # load all data from Ian's coordinate_mapping function
    print("Reading data from coordinate mapping...")
    all_data = coordinate_mapping()

    # save all data to an excel file in output folder
    output_dir = os.path.join(repo_root, 'output')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'all_data.csv')

    # save all data to the csv file
    all_data.to_csv(output_path, index=False)
    print(f"All data saved to {output_path}")

def main():
    load_data()

if __name__ == "__main__":
    main()
    