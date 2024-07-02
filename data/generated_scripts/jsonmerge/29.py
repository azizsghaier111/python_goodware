import json
from collections import OrderedDict
from jsonmerge import Merger
import numpy as np
import pytorch_lightning as pl
from unittest import mock

# JSON Merge schema
schema = {
    "properties": {
        "property1": {
            "mergeStrategy": "objectMerge",
            "mergeOptions": {
                "override": True
            }
        },
        "property2": {
            "mergeStrategy": "objectMerge",
            "mergeOptions": {
                "strategy": "$patch"
            }
        },
        "property3": {
            "mergeStrategy": "extend",
            "mergeOptions": {}
        }
    }
}

# Initialize JSON Merge object
merger = Merger(schema)

# JSON objects
base = OrderedDict([
    ("property1", {"id": 1, "value": "A"}),
    ("property2", {"id": 2, "value": "B"}),
    ("property3", [{"id": 3, "value": "C"}, {"id": 4, "value": "D"}])
])

head = OrderedDict([
    ("property1", {"id": 1, "value": "E"}),
    ("property2", {"id": 2, "value": "F"}),
    ("property3", [{"id": 5, "value": "G"}, {"id": 6, "value": "H"}])
])

merged_file_path = 'merged.json'

def write_to_json_file(data, file_path):
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def read_from_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def merge_json_objects(base_object, head_object):
    try:
        # Merge JSON objects
        result = merger.merge(base_object, head_object)

        # Save to file
        write_to_json_file(result, merged_file_path)

    except ValueError as ve:
        print(f"Merge conflict occurred: {ve}")

def main():
    base_object = read_from_json_file('base.json')
    head_object = read_from_json_file('head.json')
    merge_json_objects(base_object, head_object)

if __name__ == '__main__':
    main()