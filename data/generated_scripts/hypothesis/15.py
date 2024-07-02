# Importing necessary modules
import networkx as nx
from lark import Lark, Transformer, v_args
from hypothesis import given, note, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, Bundle
import pytest
import random

# --------------------------------------------------------------------------------------------------------
# Defining grammar for parsing
architecture = """
    start: command+
    command: "add_node" NAME 
           | "add_edge" NAME NAME 
           | "remove_node" NAME 
           | "remove_edge" NAME NAME
    NAME: LETTER+

    %import common.LETTER
    %import common.WS
    %ignore WS
"""


# --------------------------------------------------------------------------------------------------------

# Class to parse and transform commands to graph operations
class ParseToGraph(Transformer):

    def __init__(self):
        self.graph = nx.Graph()

    def start(self, commands):
        return self.graph

    def add_node(self, name):
        self.graph.add_node(str(name[0]))

    def remove_node(self, names):
        self.graph.remove_node(str(names[0]))

    def add_edge(self, names):
        self.graph.add_edge(str(names[0]), str(names[1]))

    def remove_edge(self, names):
        self.graph.remove_edge(str(names[0]), str(names[1]))

    def NAME(self, name):
        return str(name[0])


# --------------------------------------------------------------------------------------------------------

# State Machine for graph operations
class GraphStateMachine(RuleBasedStateMachine):

    # Creating bundles for nodes and edges
    Nodes = Bundle('Nodes')
    Edges = Bundle('Edges')

    def __init__(self):
        super(GraphStateMachine, self).__init__()
        self.graph = nx.Graph()
        self.parser = Lark(architecture, parser='lalr', transformer=ParseToGraph())
        self.node_list = []
        self.edge_list = []

    @rule(target=Nodes, name=st.text(min_size=1))
    def add_node(self, name):
        self.graph.add_node(name)
        self.node_list.append(name)
        return name

    @rule(node=Nodes)
    def remove_node(self, node):
        if node in self.node_list:
            self.graph.remove_node(node)
            self.node_list.remove(node)

    @rule(target=Edges, node1=Nodes, node2=Nodes)
    def add_edge(self, node1, node2):
        self.add_node(node1)
        self.add_node(node2)
        self.graph.add_edge(node1, node2)
        self.edge_list.append((node1, node2))
        return (node1, node2)

    @rule(edge=Edges)
    def remove_edge(self, edge):
        node1, node2 = edge
        if edge in self.edge_list:
            self.graph.remove_edge(node1, node2)
            self.edge_list.remove(edge)

    @precondition(lambda self: len(self.node_list) > 0)
    @rule(node1=Nodes, node2=Nodes)
    def check_edge(self, node1, node2):
        assert(self.graph.has_edge(node1, node2) == ((node1, node2) in self.edge_list))


# --------------------------------------------------------------------------------------------------------

# Testing the state machine using hypothesis generate
@given(st.data())
def test_graph_manipulation(data):
    run_state_machine_as_test(lambda: GraphStateMachine())

# --------------------------------------------------------------------------------------------------------
# Running the script as main
if __name__ == "__main__":
    # Running pytest
    pytest.main(['-v'])