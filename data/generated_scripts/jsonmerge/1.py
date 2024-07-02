import json
from collections import OrderedDict
from jsonmerge import Merger

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

try:
    # Merge JSON objects
    result = merger.merge(base, head)

    # Print merged JSON
    print(json.dumps(result, indent=4))
except ValueError as ve:
    print(f"Merge conflict occurred: {ve}")