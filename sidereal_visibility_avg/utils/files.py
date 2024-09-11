import json
from os import path


def check_folder_exists(folder_path):
    """
    Check if folder exists
    """
    return path.isdir(folder_path)

def load_json(file_path):
    """Load json file"""

    with open(file_path, 'r') as file:
        return json.load(file)
