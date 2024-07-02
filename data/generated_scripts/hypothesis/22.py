import networkx as nx
from hypothesis import given, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, rule, Bundle, initialize
import pytest
import random


MAX_INT = 5
MAX_FLOAT = 2.0

# Define the ranges for numeric generation
int_range = range(-MAX_INT, MAX_INT + 1)
float_range = st.floats(-MAX_FLOAT, MAX_FLOAT)


class DBIntegrationStateMachine(RuleBasedStateMachine):
    values = st.text()

    def __init__(self):
        super().__init__()
        self.db = {}

    @initialize(target='keys', key=st.text(min_size=1))
    def init_key(self, key):
        self.db[key] = None
        return key

    @rule(target='keys', key=st.text(min_size=1))
    def add_key(self, key):
        self.db[key] = None
        return key

    @rule(key='keys', value=values)
    def set_value(self, key, value):
        self.db[key] = value

    @rule(key='keys')
    def del_value(self, key):
        del self.db[key]

    @rule(key='keys')
    def read_value(self, key):
        return self.db.get(key)


# Function to check that the database changes are as expected
@given(st.data())
def test_db_manipulation(data):
    interpreter = data.draw(st.builds(DBIntegrationStateMachine))
    try:
        interpreter.execute()
    except AssertionError:
        pass


class NumericRangesStateMachine(RuleBasedStateMachine):

    @rule(target='integers', start=infer, stop=infer,
          step=infer)
    def integers(self, start, stop, step):
        return list(st.integers(start, stop).filter(lambda x: x % step == 0))

    @rule(target='floats', start=infer, stop=infer,
          step=infer)
    def floats(self, start, stop, step):
        return list(st.floats(start, stop).filter(lambda x: x % step < 0.0001))


# Function to check that the numeric ranges are as expected
@given(st.data())
def test_numeric_ranges_generation(data):
    interpreter = data.draw(st.builds(NumericRangesStateMachine))
    interpreter.execute()
    assert len(interpreter.integers) == len(int_range)
    assert len(interpreter.floats) == len(float_range)


if __name__ == "__main__":
    pytest.main(['-v'])