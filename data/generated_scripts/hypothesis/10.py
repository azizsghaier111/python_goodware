# Importing necessary libraries and modules
import networkx as nx
from lark import Lark, Transformer, v_args
from hypothesis import given, note, example, infer, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, rule, Bundle, initialize, consumes, precondition
from uuid import uuid4
import pytest

# Architecture of the graph
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

    def start(self, commands):
        return self.graph

    def add_node(self, name):
        # adding a node to the graph
        self.graph.add_node(str(name[0]))

    def add_edge(self, names):
        # Adding edge to the graph
        self.graph.add_edge(str(names[0]), str(names[1]))

    def NAME(self, name):
        # Returns the name of the node or edge
        return str(name[0])

# The state machine class which uses hypothesis for testing 
class GraphStateMachine(RuleBasedStateMachine):
    Nodes = Bundle('Nodes')
    Edges = Bundle('Edges')
    
    @initialize()
    def setup(self):
        # Creating an instance of the graph parser
        self.graph = nx.Graph()
        self.parser = Lark(architecture, parser='lalr', transformer=ParseToGraph())

    @rule(target=Nodes, name=st.text(min_size=1))
    def add_node(self, name):
        # Adding a node
        self.graph.add_node(name)
        return name

    @rule(name=Nodes)
    def remove_node(self, name):
        # Removing a node
        self.graph.remove_node(name)

    @rule(node1=Nodes, node2=Nodes)
    def add_edge(self, node1, node2):
        # Adding an edge
        self.graph.add_edge(node1, node2)

    @rule(node1=Nodes, node2=Nodes)
    def remove_edge(self, node1, node2):
        # Removing an edge
        if self.graph.has_edge(node1, node2):
            self.graph.remove_edge(node1, node2)

# Function for performing hypothesis testing
@given(st.data())
def test_graph_manipulation(data):
    # Creating an instance of the state machine
    interpreter = data.draw(st.builds(GraphStateMachine))
    # Running the state machine
    interpreter.execute()

# Function to check that the start and end state of the graph is the same
def test_graph_persistence():
    # Creating a start graph
    start_graph = nx.gnp_random_graph(10, 0.5, 42, directed=True)
    # Running a transformation function on our start graph
    end_graph = transform_graph(start_graph)
    # Checking if the start graph is isomorphic to the end graph
    assert nx.algorithms.isomorphism.is_isomorphic(start_graph, end_graph)

# Main function to run pytest on the given functions
if __name__ == "__main__":
    # Running pytest
    pytest.main(['-v'])