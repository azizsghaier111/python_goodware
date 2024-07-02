import networkx as nx
from lark import Lark, Transformer, v_args
from hypothesis import given, settings, example, infer, strategies as st
from hypothesis.errors import InvalidArgument
from hypothesis.stateful import RuleBasedStateMachine, rule, Bundle, initialize, consumes, precondition
from uuid import uuid4
import pytest
import socket
import ipaddress

architecture = """
    start: command+
    command: "add_node" NAME 
           | "add_edge" NAME NAME 
    NAME: LETTER+

    %import common.LETTER
    %import common.WS
    %ignore WS
"""

# Regex pattern for matching IP address
IP_ADDRESS_PATTERN = r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\." \
                     r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\." \
                     r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\." \
                     r"(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)"

# function to verify binary data
def binary_verification(binary_data):
    try:
        int(binary_data, 2)
        return True
    except ValueError:
        return False

class ParseToGraph(Transformer):
    def __init__(self):
        self.graph = nx.Graph()

    def add_node(self, name):
        self.graph.add_node(''.join(str(i) for i in name))

    def add_edge(self, names):
        self.graph.add_edge(''.join(str(i) for i in names[0]), ''.join(str(i) for i in names[1]))

    def NAME(self, name):
        return ''.join(str(i) for i in name[0])


class GraphStateMachine(RuleBasedStateMachine):
    Nodes = Bundle('NODES')
    edges_list = Bundle('EDGES')
    @initialize()
    def setup(self):
        self.graph = nx.Graph()
        self.parser = Lark(architecture, parser='lalr', transformer=ParseToGraph())


    @rule(target=Nodes, name=st.text().filter(lambda x: isinstance(x, str)))
    def add_node(self, name):
        graph_node = self.graph.add_node(name)
        return graph_node

    @rule(name=Nodes)
    def remove_node(self, name):
        self.graph.remove_node(name)

    @rule(node1=Nodes, node2=Nodes, target=edges_list)
    def add_edge(self, node1, node2):
        graph_edge = self.graph.add_edge(node1, node2)
        return graph_edge

    @rule(node1=Nodes, node2=Nodes)
    def remove_edge(self, node1, node2):
        self.graph.remove_edge(node1, node2)

# unicode strings using text()
@settings(max_examples=10)
@given(st.text(), st.binary(), st.from_regex(IP_ADDRESS_PATTERN))
def test_generators(text, binary, ip):
    if text:
        assert text.encode('utf-8') is not InvalidArgument
    if binary:
        assert type(binary) is bytes and binary_verification(binary)
    if ip and ipaddress.ip_address(ip):
        assert socket.inet_aton(ip)

# state machine
@given(st.data())
def test_graph_operations(data):
    machine = data.draw(st.builds(GraphStateMachine))
    machine.execute()

# running pytest
if __name__ == "__main__":
    pytest.main(['-v'])