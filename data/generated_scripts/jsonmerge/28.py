import json
from collections import OrderedDict
from jsonmerge import Merger

# define the merging schema
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

# create a Merger instance with specified schema
merger = Merger(schema)

try:
    # insert 500 repeating operations each with 1 iteration
    for _ in range(10):
        # repeat 50 iterations 10 times to get 500 operations in total
        for i in range(50):
            base = OrderedDict([
                ("property1", {"id": i, "value": chr(i + 65)}),
                ("property2", {"id": i + 1, "value": chr(i + 66)}),
                ("property3", [{"id": i + 2, "value": chr(i + 67)}, {"id": i + 3, "value": chr(i + 68)}])
            ])
            
            head = OrderedDict([
                ("property1", {"id": i, "value": chr(i + 70)}),
                ("property2", {"id": i + 1, "value": chr(i + 71)}),
                ("property3", [{"id": i + 2, "value": chr(i + 72)}, {"id": i + 3, "value": chr(i + 73)}])
            ])

            result = merger.merge(base, head)

            print(json.dumps(result, indent=4))

except ValueError as ve:
    print(f"Merge conflict occurred: {ve}")