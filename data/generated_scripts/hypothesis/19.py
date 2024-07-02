# Import relevant modules
# NetworkX for graph handling
import networkx as nx

# Lark for grammar parsing and transformations
from lark import Lark, Transformer, v_args

# Hypothesis for property-based testing
from hypothesis import given, note, example, infer, strategies as st

# Hypothesis stateful for state machine based testing
from hypothesis.stateful import RuleBasedStateMachine, rule, Bundle, initialize, consumes, precondition

# UUID for unique node name generation
from uuid import uuid4

# Pytest for organizing and running test cases
import pytest

# Architecture of our hypothetical graph
# Here we define the command that our graph can take
# These commands include adding a node, or adding an edge between two nodes
architecture = """
    start: command+ 
    command: "add_node" NAME 
           | "add_edge" NAME NAME 
    NAME: LETTER+ 

    %import common.LETTER 
    %import common.WS 
    %ignore WS 
"""

# Transformer class
# This class converts parsed commands from Lark into actual commands 
# that modifies the graph
class ParseToGraph(Transformer):
    def __init__(self):
        # Initializing an empty graph
        self.graph = nx.Graph()

    def start(self, commands):
        # The result of the parsing will be the final state of the graph
        return self.graph

    def add_node(self, name):
        # To add a node to the graph
        self.graph.add_node(str(name[0]))

    def add_edge(self, names):
        # To add edges to the graph
        self.graph.add_edge(str(names[0]), str(names[1]))

    def NAME(self, name):
        # Returns the name of the node or edge
        return str(name[0])

# State machine class for testing
class GraphStateMachine(RuleBasedStateMachine):
    Nodes = Bundle('Nodes')
    Edges = Bundle('Edges')

    # Whenever the class is initialized, we create a new parser and a new graph
    @initialize()
    def setup(self):
        self.graph = nx.Graph()
        self.parser = Lark(architecture, parser='lalr', transformer=ParseToGraph())
  
    @rule(target=Nodes, name=st.text(min_size=1))
    def add_node(self, name):
        # Adding a node to the graph and
        # return the name of the added node
        self.graph.add_node(name)
        return name

    @rule(name=Nodes, data=st.dictionaries(st.text(min_size=1), infer()))
    def add_data_to_node(self, name, data):
        # Adding data to an existing node
        # To make this function interesting, we will also allow adding arbitrary dictionaries as data
        # These data could be attributes of the nodes
        if name in self.graph.nodes:
            self.graph.nodes[name].update(data)

    @rule(name=Nodes)
    def remove_node(self, name):
        # Removing node from the graph,
        # We won't remove the node if it doesn't exist as it could lead to an error
        if name in self.graph.nodes:
            self.graph.remove_node(name)

    @rule(node1=Nodes, node2=Nodes)
    def add_edge(self, node1, node2):
        # This function allows adding an edge into our graph
        # If node2 doesn't exist, we won't add an edge
        if node1 != node2 and node1 in self.graph.nodes and node2 in self.graph.nodes:
            self.graph.add_edge(node1, node2)

    @rule(node1=Nodes, node2=Nodes)
    def remove_edge(self, node1, node2):
        # Removing an edge from the graph
        # We won't remove an edge if it doesn't exist as it could lead to an error
        if self.graph.has_edge(node1, node2):
            self.graph.remove_edge(node1, node2)

# Function to perform hypothesis testing
@given(st.data())
def test_graph_manipulation(data):
    # We are generating an instance of our state machine using Hypothesis
    interpreter = data.draw(st.builds(GraphStateMachine))
    # We then execute the state machine and see if any error occurs
    interpreter.execute()

def test_graph_construction():
    # Ensuring that the construction of graphs works
    graph_builder = Lark(architecture, parser='lalr', transformer=ParseToGraph())

# Main function to run the pytest on the given functions
if __name__ == "__main__":
    # Running the test suites using pytest
    pytest.main(['-v'])