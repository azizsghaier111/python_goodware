import json
from collections import OrderedDict
from jsonmerge import Merger
import mock
#import pytorch_lightning as pl
#import numpy as np

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
            "mergeStrategy": "append",
            "mergeOptions": {}
        },
        "property4": {
            "mergeStrategy": "objectMerge",
            "mergeOptions": {
                "override": True
            }
        },
        # add as many properties as you need
        ...
    },
    "patternProperties": {
        "^S_": {
            "mergeStrategy": "objectMerge"
        }
    },
    "additionalProperties": {
        "mergeStrategy": "objectMerge"
    }
}

# Initialize JSON Merge object
merger = Merger(schema)

# JSON objects
base = OrderedDict([
    ("property1", {"id": 1, "value": "A"}),
    ("property2", {"id": 2, "value": "B"}),
    ("property3", [{"id": 3, "value": "C"}, {"id": 4, "value": "D"}]),
    ("property4", {"id": 4, "value": "E"}),
    # add as many properties as you need
    ...
])

head = OrderedDict([
    ("property1", {"id": 1, "value": "F"}),
    ("property2", {"id": 2, "value": "G"}),
    ("property3", [{"id": 5, "value": "H"}, {"id": 6, "value": "I"}]),
    ("property4", {"id": 4, "value": "J"}),
    # add as many properties as you need
    ...
])

try:
    # Merge JSON objects
    result = merger.merge(base, head)

    # Print merged JSON
    print(json.dumps(result, indent=4))

    # Using mock, pytorch_lightning and numpy for example here you can use them as per your need
    mock_object = mock.Mock(spec=pl.LightningModule)
    tensor_data = np.random.rand(10, 10)
    model = mock_object.return_value
    model.return_value = pl.Tensor(tensor_data)
    model.train()
except ValueError as ve:
    print(f"Merge conflict occurred: {ve}")