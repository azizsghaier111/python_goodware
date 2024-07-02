import json
from collections import OrderedDict
from jsonmerge import Merger

# Setting up the merge schema
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

# Initial JSON objects to merge
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

try:
    # Initialized JSON Merge object
    merger = Merger(schema)

    # Merge operation
    result = merger.merge(base, head)

    # Printing the merged JSON to console
    print(json.dumps(result, indent=4))

# Handling merge conflicts
except ValueError as ve:
    print(f"Merge conflict occurred: {ve}")