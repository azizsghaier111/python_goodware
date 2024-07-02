import networkx as nx
from lark import Lark, Transformer
from hypothesis import given, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, Bundle, rule
from uuid import uuid4
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
    def __init__(self):
        self.graph = nx.Graph()

    def start(self, commands):
        return self.graph

    def add_node(self, name):
        self.graph.add_node(str(name[0]))

    def add_edge(self, names):
        self.graph.add_edge(str(names[0]), str(names[1]))

    def NAME(self, name):
        return str(name[0])

class GraphStateMachine(RuleBasedStateMachine):
    Nodes = Bundle('Nodes')

    @initialize(nodes=st.sets(st.text(min_size=1, max_size=1), min_size=1, max_size=5))
    def initialize(self, nodes):
        self.graph = nx.DiGraph()
        self.nodes = list(nodes)
        for node in self.nodes:
            self.graph.add_node(node)
        return self.graph

    @rule(target=Nodes, node1=Nodes, node2=Nodes)
    def create_edge(self, node1, node2):
        self.graph.add_edge(node1, node2)

    @rule(node=Nodes)
    def delete_node(self, node):
        if node in self.graph:
            self.graph.remove_node(node)

    @rule(node1=Nodes, node2=Nodes)
    def delete_edge(self, node1, node2):
        self.graph.remove_edge(node1, node2)

    @rule(target=Nodes)
    def create_node(self):
        new_node = str(uuid4())
        self.graph.add_node(new_node)
        return new_node

@given(st.data())
def test_graph_manipulation_with_binary_blobs(data):
    machine = GraphStateMachine()
    data.draw(machine.initialize())
    # add more interesting tests here
    # like manipulate the graph with binary blobs and check its states
    machine.execute()


@given(st.data())
def test_graph_manipulation_with_test_as_a_service(data):
    machine = GraphStateMachine()
    data.draw(machine.initialize())
    # add more interesting tests here
    # like manipulate the graph with test as a service and check its states
    machine.execute()

if __name__ == "__main__":
    pytest.main(['-v', 'BinaryBlobTests.test_binary_blob_gen', 'ServiceTests.test_service_gen'])