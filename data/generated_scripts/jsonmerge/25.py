import json
import random
from collections import OrderedDict
from jsonmerge import Merger
from unittest.mock import MagicMock

# JSON Merge Schema
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

# Merger object from JSONMerge
merger = Merger(schema)


def mock_data_properties():
    """
    Return a list of mock properties and their respective values
    """
    mock_properties = [f"property{str(i + 1)}" for i in range(random.randint(1, 100))]
    mock_values = [
        {"id": i, "value": chr(i + random.randint(65, 90))}
        for i in range(random.randint(1, 100))
    ]

    return mock_properties, mock_values


def mock_obj(mock_properties, mock_values):
    """
    Return a mocked object with defined properies and values
    """
    mock_obj = MagicMock(spec=OrderedDict)

    for prop in mock_properties:
        getattr(mock_obj, prop).return_value = mock_values

    return mock_obj


def merger_func(mocked_base, mocked_head):
    """
    Return a merged version of base and head
    """
    try:
        result = merger.merge(mocked_base, mocked_head)
        return result
    except ValueError as ve:
        print(f"Merge conflict occurred: {ve}")
        return None


def main():
    """
    Main function demonstrating the mock, merge process
    """
    # run mock and merge iteratively
    for _ in range(50):

        # generate mock data
        mock_properties, mock_values = mock_data_properties()

        # create mocked base and head objects
        mocked_base = mock_obj(mock_properties, mock_values)
        mocked_head = mock_obj(mock_properties, mock_values)

        # merge the base and head
        result = merger_func(mocked_base, mocked_head)

        # print merged result
        if result:
            print(json.dumps(result, indent=4))


if __name__ == "__main__":
    main()