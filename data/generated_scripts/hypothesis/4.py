import networkx as nx
from lark import Lark, Transformer, v_args
from hypothesis import given, strategies as st, settings, example
from hypothesis.stateful import RuleBasedStateMachine, rule, Bundle
from uuid import uuid4
import matplotlib.pyplot as plt

architecture = """
    start: command+
    command: "add_node" NAME 
           | "add_edge" NAME NAME  
    NAME: LETTER+

    %import common.LETTER
    %import common.WS
    %ignore WS
"""

class ParseToGraph(Transformer):
    def __init__(self):
        self.graph = nx.Graph()

    def start(self, commands):
        return self.graph

    def add_node(self, name):
        name = str(uuid4()) if not name else str(name[0])
        self.graph.add_node(name)
        return name

    def add_edge(self, names):  
        if len(names) >= 2:
            names = [str(uuid4()) if not name else str(name) for name in names]
            self.graph.add_edge(*names)

    def NAME(self, name):
        return str(name[0])

class GraphStateMachine(RuleBasedStateMachine):
    Nodes = Bundle('Nodes')

    @rule(target=Nodes, name=st.text(min_size=1))
    def add_node(self, name):
        self.graph.add_node(name)
        return name

    @rule(name=Nodes)
    def remove_node(self, name):
        if self.graph.has_node(name):
            self.graph.remove_node(name)

    @rule(node1=Nodes, node2=Nodes)
    def add_edge(self, node1, node2):
        self.graph.add_edge(node1, node2)

    @rule(node1=Nodes, node2=Nodes)
    def remove_edge(self, node1, node2):
        if self.graph.has_edge(node1, node2):
            self.graph.remove_edge(node1, node2)

    def __init__(self):
        super(GraphStateMachine, self).__init__()
        self.graph = nx.Graph()
        self.parser = Lark(architecture, parser='lalr', transformer=ParseToGraph())

@given(st.text().filter(lambda x: len(x) > 0))
@settings(max_examples=10)
@example("add_node")
@example("add_edge")
def test_graph_manipulation(s):
    graph_state_machine = GraphStateMachine()
    graph_state_machine.parser.parse(s)

def test_graph_persistence():
    start_graph = nx.gnp_random_graph(10, 0.5, 42, directed=False)
    end_graph = nx.algorithms.isomorphism.GraphMatcher(start_graph,start_graph)
    assert end_graph.is_isomorphic()

if __name__ == '__main__':
    import pytest

    pytest.main(['-v'])