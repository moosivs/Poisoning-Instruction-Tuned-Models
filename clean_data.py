import os
import json

def remove_key_from_json_files(folder_path, key_to_remove):
    # List all files in the given folder
    files = os.listdir(folder_path)

    # Filter only JSON files
    json_files = [file for file in files if file.endswith('.json')]

    for json_file in json_files:
        # Construct the full file path
        file_path = os.path.join(folder_path, json_file)

        # Read the JSON data
        with open(file_path, 'r') as file:
            data = json.load(file)

        # Remove the key if it exists
        if key_to_remove in data:
            del data[key_to_remove]

        # Write the updated data back to the JSON file
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)

# Example usage
folder_path = "data/nat_inst/tasks"
remove_key_from_json_files(folder_path, "Instance License")