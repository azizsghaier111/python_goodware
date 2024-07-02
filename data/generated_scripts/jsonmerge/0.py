import json
from collections import OrderedDict
from jsonmerge import Merger

# JSON Merge schema
schema = {
    "properties": {
        "yourProperty": {
            "mergeStrategy": "arrayMergeById",
            "mergeOptions": {
                "idRef": "$patch"
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

# Print merged JSON
print(json.dumps(result, indent=4))

# Mock $patch keyword in conflict scenario
schema = {
    "properties": {
        "yourProperty": {
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
    print(f"Merge conflict occurred: {ve}")