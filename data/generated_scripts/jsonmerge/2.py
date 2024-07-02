import json
import os
from collections import OrderedDict
from jsonmerge import Merger

# JSON Merge schema
schema = {
    "properties": {
        "item": {
            "mergeStrategy": "arrayMergeById",
            "mergeOptions": {
                "idRef": "/id"
            }
        }
    }
}

# Initialize JSON Merge object
merger = Merger(schema)

# JSON objects
base = OrderedDict([
    ("item1", {"id": 1, "value": "A"}),
    ("item2", {"id": 2, "value": "B"}),
    ("item3", {"id": 3, "value": "C"})
])

head = OrderedDict([
    ("item2", {"id": 2, "value": "D"}),
    ("item1", {"id": 1, "value": "E"}),
    ("item4", {"id": 4, "value": "F"})
])

# Merge JSON objects
result = merger.merge(base, head)

# Save merged JSON into a file
with open('merged.json', 'w') as f:
    f.write(json.dumps(result, indent=4))

print("\nMerged JSON has been saved into merged.json")

# Mock $patch keyword in conflict scenario
schema = {
    "properties": {
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

# JSON objects with conflict
base = {"item1": {"id": 1, "value": "A"}}
head = {"item1": {"id": 1, "value": "B"}}

try:
    # Merge JSON objects
    result = merger.merge(base, head)
except ValueError as ve:
    print(f"\nMerge conflict occurred: {ve}")

# If no conflict
else:
    # Save merged JSON into a file
    with open('merged_no_conflict.json', 'w') as f:
        f.write(json.dumps(result, indent=4))

print("\nMerged JSON with no conflict has been saved into merged_no_conflict.json")

print("\nEnd of script")