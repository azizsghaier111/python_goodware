import os
import json
import numpy as np
import torch
from pytorch_lightning import LightningModule
from unittest import mock
from jsonmerge import Merger
from collections import OrderedDict

# Create PyTorch Lightning model
class Model(LightningModule):
    def forward(self, x):
        return torch.relu(x)

# Initialize model
model = Model()

# Sets to check
mock_data = [
    [{"field1": 1, "field2": 2}, {"field2": 3, "field3": 4}],
    [{"field1": "old", "field2": "old"}, {"field1": "new", "field2": "new"}],
    [{"field1": [1,2,3]}, {"field1": [4,5,6]}],
    [{"a": 1, "b": 2, "c": 3}, {"b": 3, "c": "4", "d": 4}],
]

# Schema for wildcard patterns support
schema_wildcards = {"*": {"mergeStrategy": "overwrite"}}

# Schema for additionalProperties keyword support
schema_additionalProperties = {"additionalProperties": {"mergeStrategy": "append"}}

# Schema for $patch keyword support
schema_patch = {"$patch": {"mergeStrategy": "overwrite"}}

schemas = [schema_wildcards, schema_additionalProperties, schema_patch]

# Apply all schemas to all data synchronously, handle exceptions, and print results
results = {}
for i, schema in enumerate(schemas):
    merger = Merger(schema)
    for j, (base, head) in enumerate(mock_data):
        
        # Catch exceptions
        try:
            result = merger.merge(base, head)
        except Exception as e:
            print(f"Exception occurred: {str(e)}")
            result = None
            
        # Append results
        results.setdefault(f"Data_{j+1}", {})[f"Schema_{i+1}"] = result

# Presenting final results
for key in results.keys():
    print(f"\nFor {key}:")
    for schema in results[key].keys():
        print(f"\tUnder {schema}, merged data = {results[key][schema]}")
        
# Saving results in txt file
with open('output.txt', 'w') as file:
    for key in results.keys():
        file.write(f"\nFor {key}:\n")
        for schema in results[key].keys():
            file.write(f"\tUnder {schema}, merged data = {results[key][schema]}\n")