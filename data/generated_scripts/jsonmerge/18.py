import json
from jsonmerge import Merger
from pprint import pprint


# Configure schemas for merging with specific behaviors
schemas = {
    "wildcards": {
        "*": {
            "mergeStrategy": "overwrite"   # overwrite conflicting values
        }
    },
    "conflicts": {
        "*": {
            "mergeStrategy": "version"     # keep both versions of conflicting values
        }
    },
    "omit_field": {
        "properties": {
            "field_to_omit": {
                "mergeOptions": {
                    "ignore": True  # omit this field from the merge result
                }
            }
        },
        "*": {
            "mergeStrategy": "overwrite"
        }
    }
}

# Provide varied data sets for merging
data_sets = [
    {"field1": 1, "field2": 2, "field3": 3, "field_to_omit": "this should be gone"},
    {"field2": 3, "field3": "four", "field4": [1, 2, 3, 4], "field_to_omit": "this will be ignored"},
    {"field1": "one", "field3": "three", "field5": {"subfield1": "sub1", "subfield2": 2}},
]


# Define a function to merge and present two data sets using a particular schema
def merge_and_present(schema, data1, data2):
    merger = Merger(schema)

    try:
        merge_result = merger.merge(data1, data2)
    except ValueError as e:
        print("\nConflict encountered during merge. Error Message:")
        print(str(e))
    else:
        print(f"\nMerge result:")
        pprint(merge_result)


# Iterate through each possible combination of schema and data sets for merging
for schema_name, schema in schemas.items():

    print(f"\nAttempting to merge data sets with '{schema_name}' schema:\n")

    for i in range(len(data_sets)):

        for j in range(i+1, len(data_sets)):

            print(f"\nMerging data set {i+1} with data set {j+1}.")
            print(f"\nData set {i+1}:")
            pprint(data_sets[i])
            print(f"\nData set {j+1}:")
            pprint(data_sets[j])

            merge_and_present(schema, data_sets[i], data_sets[j])