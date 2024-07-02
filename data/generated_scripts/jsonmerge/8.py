import json
import random
from collections import OrderedDict
from jsonmerge import Merger
from unittest.mock import MagicMock

# Define the schema
schema = {
    "properties": {
        "*": {
            "mergeStrategy": "objectMerge",
            "mergeOptions": {
                "override": True
            }
        }
    }
}
 
merger = Merger(schema)

try:
    for i in range(50):
        
        # Mock base and head properties
        mockProperties = ["property{0}".format(j+1) for j in range(random.randint(0, 100))]
        mockPropertyValues = [{"id": i, "value": chr(i + random.randint(65, 90))} for _ in range(random.randint(0, 100))]

        # Mock base and head objects
        mockBase = MagicMock(spec=OrderedDict)
        mockHead = MagicMock(spec=OrderedDict)

        # Assigning properties to mocked objects
        for prop in mockProperties:
            getattr(mockBase, prop).return_value = mockPropertyValues
            getattr(mockHead, prop).return_value = mockPropertyValues

        # Merge mocked objects
        result = merger.merge(mockBase, mockHead)

        # Print the result in JSON format
        print(json.dumps(result, indent=4))
        
except ValueError as ve:
    print(f"Merge conflict occurred: {ve}")