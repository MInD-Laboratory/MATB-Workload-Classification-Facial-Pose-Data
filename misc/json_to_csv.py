import os
import json
import pandas as pd

# Construct the input and output directory paths based on the experiment number
in_directory = f"/Users/mikerichardson/Library/CloudStorage/OneDrive-SharedLibraries-MacquarieUniversity/Complexity in Action - Research/Mind and Interaction Dynamics/PNAS-MATB/openpose_json"
out_directory = f"/Users/mikerichardson/Library/CloudStorage/OneDrive-SharedLibraries-MacquarieUniversity/Complexity in Action - Research/Mind and Interaction Dynamics/PNAS-MATB/pose_data"

if not os.path.exists(out_directory):
    os.makedirs(out_directory)

# get a list of all JSON files in the input directory
json_files = [f for f in os.listdir(in_directory) if f.endswith('.json')]

# Print the number of JSON files found
num_files = len(json_files)
print(f"Found {num_files} JSON files in {in_directory}")

#order file by filename
json_files.sort()

# Iterate through all JSON files in the input directory
idx = 0
for json_file_name in json_files:
    base_name = os.path.splitext(json_file_name)[0]
    out_file_name = base_name + '.csv'
    # remove '_combined' from the output filename if it exists
    out_file_name = out_file_name.replace('_combined', '')
    out_file_path = os.path.join(out_directory, out_file_name)
    idx += 1

    # Skip if the CSV file already exists
    if os.path.exists(out_file_path):
        print(f"CSV file for {json_file_name} already exists. Skipping...")
        continue

    print(f"Processing {idx}/{num_files}: {json_file_name}...")
    json_file_path = os.path.join(in_directory, json_file_name)

    with open(json_file_path, 'r') as file:
        data = json.load(file)

    rows = []

    for frame in data:
        if "people" in frame and len(frame["people"]) > 0:
            face_keypoints = frame["people"][0]["face_keypoints_2d"]
            # Ensure that we have exactly 210 values (70 keypoints * 3 values per keypoint)
            if len(face_keypoints) == 210:
                rows.append(face_keypoints)

    if rows:
        df = pd.DataFrame(rows)

        columns = []
        for i in range(70):
            columns.extend([f'x{i+1}', f'y{i+1}', f'prob{i+1}'])
        df.columns = columns

        # Trim the DataFrame to only the last 28,800 rows (8 minutes at 60 fps)
        if len(df) > 28800:
            df = df.tail(28800)

        # Save the DataFrame as a CSV file
        df.to_csv(out_file_path, index=False)

        print(f"CSV file has been saved to {out_file_path}")
    else:
        print(f"No valid face keypoints found in {json_file_name}")
