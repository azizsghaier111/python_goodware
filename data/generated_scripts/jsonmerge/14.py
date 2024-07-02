import json
import os
from collections import OrderedDict
from jsonmerge import Merger
from unittest import mock

# Using the following libraries just for the sake of example
import numpy as np
import pytorch_lightning as pl

# Define a JSON Merge schema
schema = {
    "properties": {
        "item": {
            "mergeStrategy": "objectMerge",
            "mergeOptions": {
                "strategy": "discard"
            }
        }
    }
}

merger = Merger(schema)


def perform_merge(base, head):
    try:
        # Merge JSON objects
        result = merger.merge(base, head)
        print("\nMerged JSON is: ")
        print(json.dumps(result, indent=4))
        return result
    except ValueError as ve:
        print(f"\nMerge conflict occurred: {ve}")


def save_merged_json(file_path, merged_json):
    # Save merged JSON into a file
    with open(file_path, 'w') as f:
        f.write(json.dumps(merged_json, indent=4))
        print(f"\nMerged JSON has been saved into {file_path}")


# JSON elements
base_elements = OrderedDict([
    ("item1", {"id": 1, "value": "A"}),
    ("item2", {"id": 2, "value": "B"}),
    ("item3", {"id": 3, "value": "C"})
])

head_elements = OrderedDict([
    ("item4", {"id": 4, "value": "D"}),
    ("item1", {"id": 1, "value": "E"}),
    ("item2", {"id": 2, "value": "F"})
])

mock_base = mock.Mock()
mock_base.side_effect = base_elements

mock_head = mock.Mock()
mock_head.side_effect = head_elements

merged_json = perform_merge(mock_base(), mock_head())
save_merged_json('merged.json', merged_json)

# JSON elements with conflict
base_conf = {"item1": {"id": 1, "value": "A"}}
head_conf = {"item1": {"id": 1, "value": "B"}}

mock_base.side_effect = base_conf
mock_head.side_effect = head_conf

merged_no_conflict_json = perform_merge(mock_base(), mock_head())
save_merged_json('merged_no_conflict.json', merged_no_conflict_json)

print("\nEnd of script")