import json
from os import path
from glob import glob
from concurrent.futures import ThreadPoolExecutor

def check_folder_exists(folder_path):
    """
    Check if folder exists
    """

    return path.isdir(folder_path)

def load_json(file_path):
    """Load json file"""

    with open(file_path, 'r') as file:
        return json.load(file)

def read_mapping(mapping_folder):
    """
    Read mapping with multi-threads
    """

    # Get the list of JSON files
    json_files = glob(path.join(mapping_folder, "*.json"))

    # Load JSON files in parallel
    total_map = {}
    with ThreadPoolExecutor() as executor:
        for result in executor.map(load_json, json_files):
            total_map.update(result)

    # Convert keys and values to integers and sort
    total_map = {int(k): v for k, v in total_map.items()}
    sorted_total_map = dict(sorted(total_map.items()))

    indices = list(sorted_total_map.keys())
    ref_indices = list(sorted_total_map.values())

    return indices, ref_indices