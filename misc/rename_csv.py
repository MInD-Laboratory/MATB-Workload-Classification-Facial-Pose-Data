import os
import json
import pandas as pd


dir = f"/Users/mikerichardson/Library/CloudStorage/OneDrive-SharedLibraries-MacquarieUniversity/Complexity in Action - Research/Mind and Interaction Dynamics/PNAS-MATB/pose_data"

# get a list of all JSON files in the input directory
csv_files = [f for f in os.listdir(dir) if f.endswith('.csv')]

# Print the number of JSON files found
num_files = len(csv_files)
print(f"Found {num_files} CSV files in {dir}")

#order file by filename
csv_files.sort()

# Iterate through all csv files in the input directory
#rename them by adding '_pose' before .csv in filename
for csv_file_name in csv_files:
    base_name = os.path.splitext(csv_file_name)[0]
    new_file_name = base_name + '_pose.csv'
    old_file_path = os.path.join(dir, csv_file_name)
    new_file_path = os.path.join(dir, new_file_name)
    print(f"Renaming {csv_file_name} to {new_file_name}...")
    os.rename(old_file_path, new_file_path)
