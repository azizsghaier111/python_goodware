import json
from collections import OrderedDict
from jsonmerge import Merger
from unittest.mock import Mock
import pytorch_lightning as pl
import numpy as np

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
            "mergeOptions": {"strategy": "$patch"}
        },
        "property3": {
            "mergeStrategy": "arrayMergeById",
            "mergeOptions": {"idRef": "id"}
        }
    }
}

merger = Merger(schema)

def create_data(i):
    # Return base and head data.
    data = OrderedDict([("property1", {"id": i, "value": chr(i + 65)}),
                            ("property2", {"id": i + 1, "value": chr(i + 66)}),
                            ("property3", [{"id": i + 2, "value": chr(i + 67)}, {"id": i + 3, "value": chr(i + 68)}])])
    return data, data

try:
    for i in range(50):
        base, head = create_data(i)
        result = merger.merge(base, head)
        print(json.dumps(result, indent=4))

except ValueError as ve:
    print(f"Merge conflict occurred: {ve}")