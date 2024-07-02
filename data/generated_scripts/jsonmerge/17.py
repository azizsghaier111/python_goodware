import json
from jsonmerge import Merger
# Additionally importing pprint for neat console outputs
from pprint import pprint

# Defining multiple JSON schemata to use in merging operations
schemas = {
    "wildcards": { "*": { "mergeStrategy": "overwrite" } },
    "conflicts": { "*": { "mergeStrategy": "version" } },
    "omit_field": {
        "properties": { "field_to_omit": { "mergeOptions": { "ignore": True } } },
        "*": { "mergeStrategy": "overwrite" }
    }
}

# Defining a few basic data sets
# Will be reusing them during the different merge operations
data_sets = [
    { "field1": 1, "field2": 2, "field3": 3, "field_to_omit": "this should be gone" },
    { "field2": 3, "field3": "four", "field4": [1,2,3,4], "field_to_omit": "this should be ignored" },
    { "field1": "one", "field3": "three", "field5": { "subfield1": "sub1", "subfield2": 2 } }
]

# Perform merge with each schema and each data combination
for schema_name, schema in schemas.items():
    print(f"\nMerging with schema: {schema_name}")
    for i in range(len(data_sets)):
        for j in range(i+1, len(data_sets)):
            print(f"\nMerging data set {i+1} and {j+1}:")

            # Print data before merge
            print("\nData before merge:")
            print(f"Set {i+1}:")
            pprint(data_sets[i])
            print(f"Set {j+1}:")
            pprint(data_sets[j])

            # Perform the merge operation
            merger = Merger(schema)
            try:
                result = merger.merge(data_sets[i], data_sets[j])
            except ValueError as e:
                print("\nMerge operation resulted in a conflict:")
                print(e)
                continue
            print("\nData after merge:")
            pprint(result)