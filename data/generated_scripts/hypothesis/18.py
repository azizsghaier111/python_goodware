import random
from uuid import uuid4
from lark import Lark, Transformer, v_args
from hypothesis import given, strategies as st, settings, example
from hypothesis.stateful import RuleBasedStateMachine, rule, Bundle
import networkx as nx

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
        super().__init__()
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
        self._execute_command(f"add_node {name}")

    @rule(name=Nodes)
    def remove_node(self, name):
        if name in self.graph:
            self.graph.remove_node(name)

    @rule(node1=Nodes, node2=Nodes)
    def add_edge(self, node1, node2):
        if node1 in self.graph and node2 in self.graph:
            self._execute_command(f"add_edge {node1} {node2}")

    @rule(node1=Nodes, node2=Nodes)
    def remove_edge(self, node1, node2):
        if self.graph.has_edge(node1, node2):
            self.graph.remove_edge(node1, node2)

    def __init__(self):
        super(GraphStateMachine, self).__init__()
        self.graph = nx.DiGraph()
        self.parser = Lark(architecture, parser='lalr', transformer=ParseToGraph())
        self.commands = []

    def _execute_command(self, command):
        self.commands.append(command)
        self.last_graph = self.graph.copy()
        self.graph = self.parser.parse(' '.join(self.commands))

    def teardown(self):
        assert nx.is_isomorphic(self.last_graph, self.graph)

@given(st.lists(st.text(min_size=1), min_size=1, max_size=100))
@settings(max_examples=10)
@example(["add_node A"])
@example(["add_edge A B"])
def test_graph_manipulation(commands):
    graph_state_machine = GraphStateMachine()
    for command in commands:
        graph_state_machine._execute_command(command)