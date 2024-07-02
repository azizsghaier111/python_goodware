@given(some_data=strategies.dictionaries(keys=strategies.text(), values=strategies.text()))
def test_some_func(some_data):
    result = some_func(some_data)
    # assert something based on your falsification test logic or conditions