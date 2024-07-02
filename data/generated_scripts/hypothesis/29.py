from hypothesis import given, strategies, settings, HealthCheck
from networkx.algorithms.iso import is_isomorphic
from networkx import ordered
from lark import Lark, Transformer

settings.register_profile("ci", suppress_health_check=(HealthCheck.too_slow,))


class DictToLower(Transformer):
    def text(self, items):
        return items[0].lower()

    def __init__(self):
        self.count = 0

    # The dict function uses the items variable
    def dict(self, items):
        items_list=list(items)
        dictionary={}

        for i in items_list:
            self.count += 1  # self.count keeps track of the loop iterations
            temp_dictionary = {f'key_{self.count}': f'value_{self.count}'}

            # Dict comprehension is used for generating dictionary with numbered keys and values
            dictionary = {**dictionary, **temp_dictionary}
            
        return dictionary
        

def process(data):
    parser = Lark('start: dict', start='start')    
    tree = parser.parse(data)
    return DictToLower().transform(tree)


@given(some_data=strategies.dictionaries(keys=strategies.text(), values=strategies.text()))
def test_process(some_data):
    dict_str = str(some_data)
    result = process(dict_str)
    falsified = not all(k.islower() and v.islower() for k, v in result.items())
    
    assert falsified == False


@given(some_data=strategies.dictionaries(keys=strategies.text(), values=strategies.text()))
def test_generate_input(some_data):
    assert isinstance(some_data, dict)
    keys = [k for k in some_data.keys()]
    values = [v for v in some_data.values()]

    assert all(isinstance(k, str) for k in keys)
    assert all(isinstance(v, str) for v in values)


@given(test_range=strategies.integers())
def test_generate_range(test_range):
    assert isinstance(test_range, int)


@given(node_data1=strategies.integers(), node_data2=strategies.integers())
def test_graph_equivalence(node_data1, node_data2):
    assert isinstance(node_data1, int)
    assert isinstance(node_data2, int)

    G1 = ordered.OrderedDiGraph()
    G2 = ordered.OrderedDiGraph()

    G1.add_edge(node_data1, node_data2)
    G2.add_edge(node_data1, node_data2)

    assert is_isomorphic(G1, G2) == True


if __name__ == "__main__":
   pytest.main()