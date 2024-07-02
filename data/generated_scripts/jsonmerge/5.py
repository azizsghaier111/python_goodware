import os
import json
import numpy as np
from unittest import mock
import pytorch_lightning as pl
from collections import OrderedDict
from jsonmerge import Merger

# JSON Merge schema
schema = {
    "properties": {
    "id": {
        "type": "string"
    },
    "value": {
        "type": "string"
    },
    "item": {
        "mergeStrategy": "arrayMergeById",
        "mergeOptions": {
            "idRef": "/id"
        }
    }
    }
}

# Mock method to generate random values
def mock_value_generator():
    return ''.join(np.random.choice(['A', 'B', 'C', 'D', 'E', 'F'], size=10))


mock_value_gen = mock.MagicMock(side_effect=mock_value_generator)

# Initialize JSON Merge object
merger = Merger(schema)

# Generate mock JSON objects
base = OrderedDict()
for i in range(1, 51):
    base["item"+str(i)] = {"id": i, "value": mock_value_gen()}

head = OrderedDict()
for i in range(50, 0, -1):
    head["item"+str(i)] = {"id": i, "value": mock_value_gen()}

# Merge JSON objects
result = merger.merge(base, head)

# Save merged JSON into a file
with open('merged.json', 'w') as f:
    json.dump(result, f, indent=4)

print("\nMerged JSON has been saved into merged.json")

# Mock the $patch keyword in conflict scenario

schema = {
    "properties": {
    "id": {
        "type": "string"
    },
    "value": {
        "type": "string"
    },
    "item": {
        "mergeStrategy": "objectMerge",
        "mergeOptions": {
            "strategy": "$patch"
        }
    }
    }
}

# Initialize JSON Merge object
merger = Merger(schema)

# Generate mock JSON objects with conflict
base = {"item1": {"id": 1, "value": mock_value_gen()}}
head = {"item1": {"id": 1, "value": mock_value_gen()}}

try:
    # Merge JSON objects
    result = merger.merge(base, head)
except ValueError as ve:
    print(f"\nMerge conflict occurred: {ve}")

else:
    # Save merged JSON into a file
    with open('merged_no_conflict.json', 'w') as f:
        json.dump(result, f, indent=4)

print("\nMerged JSON with no conflict has been saved into merged_no_conflict.json")

print("\nEnd of script")