import networkx as nx
from lark import Lark, Transformer, v_args
from hypothesis import given, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, rule, Bundle
import pytest

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
    def add_node(self, name):
        self.graph.add_node(str(name[0]))

    def add_edge(self, names):
        self.graph.add_edge(str(names[0]), str(names[1]))

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
        self.graph.remove_node(name)

    @rule(node1=Nodes, node2=Nodes, _uuid=st.uuids())
    def add_edge(self, node1, node2, _uuid):
        self.graph.add_edge(node1, node2, uuid=_uuid)

    @rule(name=Nodes)
    def remove_edge_by_name(self, name):
        edges = list(self.graph.edges(name))

        for edge in edges:
            self.graph.remove_edge(*edge)

def test_graph_manipulation():
    @given(st.data())
    def run_test(data):
        interpreter = data.draw(st.builds(GraphStateMachine))
        interpreter.execute()

def test_graph_persistence():
    start_graph = nx.gnp_random_graph(10, 0.5, 42, directed=True)
    end_graph = transform_graph(start_graph)
    assert nx.algorithms.isomorphism.is_isomorphic(start_graph, end_graph)

if __name__ == '__main__':
    import pytest
    pytest.main(['-v'])