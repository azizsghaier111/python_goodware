# Importing necessary libraries and modules
import networkx as nx
from lark import Lark, Transformer, v_args
from hypothesis import given, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, rule, Bundle, initialize
from uuid import uuid4
import pytest


# Architecture of the graph
# This grammar defines commands to add nodes and edges to a graph
architecture = """
    start: command+
    command: "add_node" NAME 
           | "add_edge" NAME NAME 
    NAME: LETTER+

    %import common.LETTER
    %import common.WS
    %ignore WS
"""


# Transformer class for converting parsed commands into graph manipulations
class ParseToGraph(Transformer):
    def __init__(self):
        self.graph = nx.Graph()
    
    # The 'start' method keeps track of our graph's state
    def start(self, commands):
        return self.graph

    # 'add_node' method adds a new node to the graph
    def add_node(self, name):
        self.graph.add_node(str(name[0]))

    # 'add_edge' method creates an edge between two nodes
    def add_edge(self, names):
        self.graph.add_edge(str(names[0]), str(names[1]))

    # 'NAME' method simply returns the name of the node/edge
    def NAME(self, name):
        return str(name[0])


# The state machine class which uses hypothesis for testing 
class GraphStateMachine(RuleBasedStateMachine):
    Nodes = Bundle('Nodes')
    Edges = Bundle('Edges')
    
    # Initialization method
    @initialize()
    def setup(self):
        # Instantiating the graph and parser
        self.graph = nx.Graph()
        self.parser = Lark(architecture, parser='lalr', transformer=ParseToGraph())

    # 'add_node' method to add a node to the graph
    @rule(target=Nodes, name=st.text(min_size=1))
    def add_node(self, name):
        self.graph.add_node(name)
        return name

    # 'remove_node' method to remove a node from the graph
    @rule(name=Nodes)
    def remove_node(self, name):
        self.graph.remove_node(name)

    # 'add_edge' method to add an edge between two nodes
    @rule(node1=Nodes, node2=Nodes)
    def add_edge(self, node1, node2):
        self.graph.add_edge(node1, node2)

    # 'remove_edge' method to remove an edge between two nodes
    @rule(node1=Nodes, node2=Nodes)
    def remove_edge(self, node1, node2):
        if self.graph.has_edge(node1, node2):
            self.graph.remove_edge(node1, node2)


# Function for performing hypothesis testing
@given(st.data())
def test_graph_manipulation(data):
    # Instantiating state machine
    interpreter = data.draw(st.builds(GraphStateMachine))
    # Executing state machine
    interpreter.execute()

if __name__ == "__main__":
    # Running pytest
    pytest.main(['-v'])