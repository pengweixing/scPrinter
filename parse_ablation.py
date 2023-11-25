import json
import sys

def read_json_file(file_path):
    """Read and parse a JSON file."""
    with open(file_path, 'r') as file:
        return json.load(file)

def extract_data(json_data):
    """Extract data from the JSON object starting from 'savename'."""
    keys_to_extract = list(json_data.keys())
    return {key: json_data[key] for key in keys_to_extract}

def create_tsv_row(data_dict):
    """Create a TSV formatted string from a dictionary."""
    return '\t'.join(str(data_dict[key]) for key in sorted(data_dict.keys()))

def process_files(file_paths):
    """Process a list of JSON files and return a TSV formatted string."""
    tsv_rows = []
    for path in file_paths:
        fn = path.split("/")[-1].split(".")[0]
        json_data = read_json_file(path)
        extracted_data = extract_data(json_data)
        if len(tsv_rows) == 0:
            tsv_rows.append('id\t'+'\t'.join(sorted(list(json_data.keys()))))
        tsv_rows.append(fn+'\t'+create_tsv_row(extracted_data))
    return '\n'.join(tsv_rows)

if __name__ == "__main__":
    # Assuming the script is called with JSON file paths as arguments
    file_paths = sys.argv[1:]
    tsv_data = process_files(file_paths)
    print(tsv_data)
