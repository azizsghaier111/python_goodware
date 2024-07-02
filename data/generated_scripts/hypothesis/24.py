# Necessary Imports for Testing and Graph Management
import networkx as nx 
from lark import Lark, Transformer, v_args
from hypothesis import given, note, example, infer, strategies as st
import logging
import pytest
from hypothesis.stateful import RuleBasedStateMachine, rule, Bundle, initialize, consumes, precondition
import matplotlib.pyplot as plt

# Parser architecture definition
architecture = """
    start: command+
    command: "add_node" NAME 
           | "add_edge" NAME NAME 
    NAME: LETTER+

    %import common.LETTER
    %import common.WS
    %ignore WS
"""

# Transformer class definition
class ParseToGraph(Transformer):
    # Constructor: Graph Initialzation
    def __init__(self): 
        self.graph = nx.Graph() 

    # Start of the parser function
    def start(self, commands): 
        return self.graph 

    # Add node command
    def add_node(self, name): 
        self.graph.add_node(str(name[0]))

    def add_edge(self, names): 
        self.graph.add_edge(str(names[0]), str(names[1]))

    def NAME(self, name): 
        return str(name[0]) 
        
# The state machine class for testing
class GraphStateMachine(RuleBasedStateMachine):
    Nodes = Bundle('Nodes')

    @initialize()
    def init(self):
        self.graph = nx.Graph()
        self.parser = Lark(architecture, parser='lalr', transformer=ParseToGraph())

    @rule(target=Nodes, name=st.text(min_size=1))
    def add_node(self, name):
        self.graph.add_node(name)
        return name

    @rule(node1=Nodes, node2=Nodes)
    def add_edge(self, node1, node2):
        self.graph.add_edge(node1, node2)

 # Function to draw the graph
def draw_graph(G):
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos, edge_color='r', arrows = True)
    plt.show()

## Function that returns a generator and a list
@given(st.lists(st.integers()))
def test_list_and_generator_are_same(lst):
    def generator():
        return (i for i in lst)

    assert list(generator()) == lst

## Begin Hypothesis Testing 
@given(st.data())
def test_stateful_graph_manipulation(data):
    interpreter = data.draw(st.builds(GraphStateMachine))
    interpreter.execute()

if __name__ == "__main__":
    pytest.main(['-v']) # Run pytest framework