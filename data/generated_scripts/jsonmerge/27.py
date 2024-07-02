import json
from jsonmerge import Merger
from pprint import pprint
import mock
import pytorch_lightning as pl
import numpy as np

# Define more complex schemas for merging
schemas = {
    "wildcards": {
        "*": {
            "mergeStrategy": "overwrite"
        }
    },
    "conflicts": {
        "*": {
            "mergeStrategy": "version"
        }
    },
    "omit_field": {
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
}

# Provide more complex and varied data sets for merging
data_sets = [
    {"field1": 1, "field2": 2, "field3": 3, "field_to_omit": "this should be gone"},
    {"field2": 3, "field3": "four", "field4": [1, 2, 3, 4], "field_to_omit": "this will be ignored"},
    {"field1": "one", "field3": "three", "field5": {"subfield1": "sub1", "subfield2": 2}},
    {"field5": {"subfield1": "different sub1", "subfield3": 3}, "field6": "six"},
    {"field1": 111, "field3": 333, "field7": [7, 77, 777]},
]

# Function to print data set
def print_data_set(dataset_number, dataset):
    print(f"\nData set {dataset_number}:")
    pprint(dataset)

# Function to merge and present two data sets under a particular schema
def merge_and_present(schema, data1, data2):
    merger = Merger(schema)
    try:
        merge_result = merger.merge(data1, data2)
    except ValueError as e:
        print("\nConflict encountered during merge. Error Message:")
        print(str(e))
    else:
        print("\nMerge result:")
        pprint(merge_result)

# Iterate through all possible combinations of schema and data sets for merging
for schema_name, schema in schemas.items():
    print(f"\n============================================")
    print(f"Attempting to merge data sets with '{schema_name}' schema:\n")
    for i in range(len(data_sets)):
        for j in range(i + 1, len(data_sets)):
            print(f"\n++++++++++++++++++++++++++++++")
            print(f"Merging data set {i + 1} with data set {j + 1}.")
            print_data_set(i + 1, data_sets[i])
            print_data_set(j + 1, data_sets[j])
            merge_and_present(schema, data_sets[i], data_sets[j])
            print(f"\n++++++++++++++++++++++++++++++")
    print(f"============================================")