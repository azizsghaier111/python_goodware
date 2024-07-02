import os
import json
from collections import OrderedDict
from unittest import mock
import numpy as np
from jsonmerge import Merger

# Define the more sophisticated schema
schema_with_wildcards = {
    "*": {
        "mergeStrategy": "overwrite"
    }
}

schema_with_conflicts = {
    "*": {
        "mergeStrategy": "version"
    }
}

schema_with_omit = {
    "properties": {
        "field_to_omit": {
            "mergeOptions": {
                "ignore": True
            }
        }
    },
    "*": {
        "mergeStrategy": "overwrite"
    }
}

# Get some base and head data
base = {"field1": 1, "field2": 2, "field3": 3, "field_to_omit": "this should be gone"}
head = {"field2": 3, "field3": "4", "field4": [1,2,3,4], "field_to_omit": "this should be ignored"}

# Use the wildcard strategy - any explicitly mentioned field in head overwrites base
merger = Merger(schema_with_wildcards)
result = merger.merge(base, head)
print(result) # {'field1': 1, 'field2': 3, 'field3': '4', 'field4': [1,2,3,4], 'field_to_omit': 'this should be ignored'}

# Throw an exception for conflict
merger = Merger(schema_with_conflicts)
try:
    result = merger.merge(base, head)
except: # Any exception necessary, put specifics if needed
    print("There was a conflict") # Throws ValueError

# Ignore merging of one field
merger = Merger(schema_with_omit)
result = merger.merge(base, head)
print(result) # {'field1': 1, 'field2': 3, 'field3': '4', 'field4': [1,2,3,4]}