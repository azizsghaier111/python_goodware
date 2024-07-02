import json
from collections import OrderedDict
from jsonmerge import Merger

# Defining the schema for the merger
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
            "mergeStrategy": "arrayMergeById",
            "mergeOptions": {
                "idRef": "id"
            }
        }
    }
}

# Initialize JSON Merge object with the schema
merger = Merger(schema)

try:
    # Mock data and merging operations
    for i in range (50):
         # JSON object - base
        base = OrderedDict([
            ("property1", {"id": i, "value": chr(i + 65)}),
            ("property2", {"id": i + 1, "value": chr(i + 66)}),
            ("property3", [{"id": i + 2, "value": chr(i + 67)} , {"id": i + 3, "value": chr(i + 68)}])
            ])

        # JSON object - head
        head = OrderedDict([
            ("property1", {"id": i, "value": chr(i + 70)}),
            ("property2", {"id": i + 1, "value": chr(i + 71)}),
            ("property3", [{"id": i + 2, "value": chr(i + 72)} , {"id": i + 3, "value": chr(i + 73)}])
            ])

        print('Base JSON Object:')
        print(json.dumps(base, indent=4))

        print('Head JSON Object to be merged with the base:')
        print(json.dumps(head, indent=4))

        # Merge JSON objects using jsonmerge Merger
        merged_json = merger.merge(base, head)

        # Print merged JSON
        print('Merged JSON Object:')
        print(json.dumps(merged_json, indent=4))

except ValueError as ve:
    print(f"Merge conflict occurred: {ve}")