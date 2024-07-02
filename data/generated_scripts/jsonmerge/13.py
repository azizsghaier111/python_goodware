import json
from collections import OrderedDict
from jsonmerge import Merger

# JSON Merge schema for 'arrayMergeById' strategy
schema_array_merge_by_id = {
    "properties": {
        "yourProperty": {
            "mergeStrategy": "arrayMergeById",
            "mergeOptions": {
                "idRef": "$patch"
            }
        }
    }
}

# JSON Merge schema for 'objectMerge' strategy
schema_object_merge = {
    "properties": {
        "yourProperty": {
            "mergeStrategy": "objectMerge",
            "mergeOptions": {
                "strategy": "$patch"
            }
        }
    }
}

# JSON Merge schema for 'version' strategy
schema_version = {
    "properties": {
        "yourProperty": {
            "mergeStrategy": "version",
            "mergeOptions": {
                "strategy": "$patch"
            }
        }
    }
}

# JSON Merge schema for 'overwrite' strategy
schema_overwrite = {
    "properties": {
        "yourProperty": {
            "mergeStrategy": "overwrite",
            "mergeOptions": {
                "strategy": "$patch"
            }
        }
    }
}

# Array of schemas
schemas = [schema_array_merge_by_id, schema_object_merge, schema_version, schema_overwrite]

# JSON objects for merging
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

# For each schema, merge JSON objects and print result
for schema in schemas:
    # Initialize JSON Merge object
    merger = Merger(schema)
    
    try:
        # Merge JSON objects
        result = merger.merge(base, head)
        # Print merged JSON
        print(json.dumps(result, indent=4))
    except ValueError as ve:
        print(f"Merge conflict occurred: {ve}")

# JSON objects with conflict for 'objectMerge' strategy
base_conflict = {"item1": {"id": 1, "value": "A"}}
head_conflict = {"item1": {"id": 1, "value": "B"}}

merger_conflict = Merger(schema_object_merge)

try:
    # Merge JSON objects
    result_conflict = merger_conflict.merge(base_conflict, head_conflict)
except ValueError as ve:
    print(f"Merge conflict occurred: {ve}")