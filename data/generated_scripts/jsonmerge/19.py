import json
from collections import OrderedDict
from jsonmerge import Merger

# JSON Merge schemas for different strategies
schemas = {
    'arrayMergeById': {
        "properties": {
            "yourProperty": {
                "mergeStrategy": "arrayMergeById",
                "mergeOptions": {
                    "idRef": "$patch"
                }
            }
        }
    },
    'objectMerge': {
        "properties": {
            "yourProperty": {
                "mergeStrategy": "objectMerge",
                "mergeOptions": {
                    "strategy": "$patch"
                }
            }
        }
    },
    'version': {
        "properties": {
            "yourProperty": {
                "mergeStrategy": "version",
                "mergeOptions": {
                    "strategy": "$patch"
                }
            }
        }
    },
    'overwrite': {
        "properties": {
            "yourProperty": {
                "mergeStrategy": "overwrite",
                "mergeOptions": {
                    "strategy": "$patch"
                }
            }
        }
    }
}

# JSON objects for merging
data = {
    'base': OrderedDict([
        ("item1", {"id": 1, "value": "A"}),
        ("item2", {"id": 2, "value": "B"}),
        ("item3", {"id": 3, "value": "C"})
    ]),
    'head': OrderedDict([
        ("item2", {"id": 2, "value": "D"}),
        ("item1", {"id": 1, "value": "E"}),
        ("item4", {"id": 4, "value": "F"})
    ])
}

# Iterate on different schemas and data to merge
for s in schemas:
    schema = schemas[s]
    for d in data:
        base = data[d]
        # Take other data for head
        head = data[list(set(data.keys()) - {d})[0]]

        # Initialize JSON Merge object
        merger = Merger(schema)

        try:
            # Merge JSON objects
            result = merger.merge(base, head)
            print("Merging base and head using '{0}' strategy successful. Merged JSON:".format(s))
            print(json.dumps(result, indent=4))

        except ValueError as ve:
            print("Merge conflict occurred using '{0}' strategy: {1}".format(s, ve))

# JSON objects with conflict for 'objectMerge' strategy
base_object_conflict = {"item1": {"id": 1, "value": "A"}}
head_object_conflict = {"item1": {"id": 1, "value": "B"}}

merger_object_conflict = Merger(schemas['objectMerge'])

try:
    result_conflict = merger_object_conflict.merge(base_object_conflict, head_object_conflict)
except ValueError as ve:
    print("Merge conflict occurred using 'objectMerge' strategy: {0}".format(ve))