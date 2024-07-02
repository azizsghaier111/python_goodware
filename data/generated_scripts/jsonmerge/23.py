import json
from collections import OrderedDict
from jsonmerge import Merger

# Here, we are defining the JSON Merge schema
schema = {
    "properties": {
        "identity": {
            "mergeStrategy": "objectMerge",
            "mergeOptions": {
                "override": True
            }
        },
        "personalDetail": {
            "mergeStrategy": "objectMerge",
            "mergeOptions": {
                "strategy": "$patch"
            }
        },
        "projects": {
            "mergeStrategy": "extend",
            "mergeOptions": {}
        }
    }
}

# Now, we initialize the JSON Merge object
merger = Merger(schema)

# Let's define two complex JSON objects

# The base JSON
base = OrderedDict([
    ("identity", {"id": 1, "name": "John Doe", "email": "johndoe@example.com"}),
    ("personalDetail", {"dob": "1980-01-01", "country": "USA"}),
    ("projects", [{
        "id": 101,
        "title": "Project A",
        "description": "Description for Project A",
        "startDate": "2019-01-01",
        "endDate": "2020-01-01"
    }, {
        "id": 102,
        "title": "Project B",
        "description": "Description for Project B",
        "startDate": "2020-02-01",
        "endDate": "2021-02-01"
    }])
])

# The head JSON that will overwrite some values in the base JSON
head = OrderedDict([
    ("identity", {"id": 1, "name": "John D.", "email": "johndoeupdated@example.com"}),
    ("personalDetail", {"dob": "1980-01-01", "country": "Canada"}),
    ("projects", [{
        "id": 103,
        "title": "Project C",
        "description": "Description for Project C",
        "startDate": "2021-03-01",
        "endDate": "2022-03-01"
    }, {
        "id": 104,
        "title": "Project D",
        "description": "Description for Project D",
        "startDate": "2022-04-01",
        "endDate": "2023-04-01"
    }])
])

# Here we are defining the merge_json function that will handle the merging of the given JSON objects
def merge_json(base_json, head_json):
    try:
        # Merge base JSON and head JSON
        result = merger.merge(base_json, head_json)

        # Pretty print the merged result
        print(json.dumps(result, indent=4))

    # Handle both the ValueError and KeyError Exception
    except (ValueError, KeyError) as err:
        print(f"An error occurred during the merge process: {err}")

# Define the main function
def main(base_json, head_json):
    # Call the merge_json function to merge the base and head JSON objects
    merge_json(base_json, head_json)

# Ensuring this script is not imported as a module, by checking if this is the main script being run
if __name__ == "__main__":
    # Call the main function
    main(base, head)