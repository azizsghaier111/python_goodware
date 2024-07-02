import networkx as nx
from lark import Lark, Transformer, v_args
from hypothesis import given, note, example, infer, strategies as st
from hypothesis.stateful import RuleBasedStateMachine, rule, Bundle, initialize, consumes, precondition
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

# State Machine
class GraphStateMachine(RuleBasedStateMachine):
    Nodes = Bundle('Nodes')
    Edges = Bundle('Edges')
    
    @initialize()
    def setup(self):
        self.graph = nx.Graph()
        self.data = ""
        self.parser = Lark(architecture, parser='lalr', transformer=ParseToGraph())

    @rule(target=Nodes, name=st.text(min_size=1))
    def add_node(self, name):
        self.graph.add_node(name)
        self.data += "add_node " + name + "\n"
        return name

    @rule(name=Nodes)
    def remove_node(self, name):
        self.graph.remove_node(name)
        self.data = self.data.replace("add_node " + name + "\n", "")
        # remove all edge related to this node
        self.data = '\n'.join([line for line in self.data.split("\n") if not line.endswith(name)])

    @rule(node1=Nodes, node2=Nodes)
    def add_edge(self, node1, node2):
        self.data += "add_edge " + node1 + " " + node2 + "\n"
        self.graph.add_edge(node1, node2)

    @rule(node1=Nodes, node2=Nodes)
    def remove_edge(self, node1, node2):
        if self.graph.has_edge(node1, node2):
            self.graph.remove_edge(node1, node2)
            self.data = self.data.replace("add_edge " + node1 + " " + node2 + "\n", "")

    @precondition(lambda self: len(self.data.split("\n"))>0)
    @rule()
    def process(self):
        tree = self.parser.parse(self.data)
        processed_graph = ParseToGraph().transform(tree)
        
        assert nx.is_isomorphic(self.graph, processed_graph)

# Settings for the Hypothesis testing
@given(st.data())
def test_graph_manipulation(data):
    data.draw(st.builds(GraphStateMachine))

# Function to check that the start and end state of the graph is the same
def test_graph_persistence():
    start_graph = nx.gnp_random_graph(10, 0.5, 42, directed=True)
    end_graph = transform_graph(start_graph)
    assert nx.algorithms.isomorphism.is_isomorphic(start_graph, end_graph)

if __name__ == "__main__":
    pytest.main(['-v'])