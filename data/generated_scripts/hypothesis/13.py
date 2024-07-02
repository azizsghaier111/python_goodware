import networkx as nx
from lark import Lark, Transformer
from hypothesis import given, strategies as st, settings
from hypothesis.stateful import RuleBasedStateMachine, rule

GRAPH_ACTION_STRATEGY = st.from_regex(r"^(add_node [a-z]$)|(add_edge [a-z] [a-z]$)", fullmatch=True)


class GraphMachine(RuleBasedStateMachine):
    def __init__(self):
        super().__init__()
        self.parser = Lark(graph_grammar, parser='lalr', transformer=TreeToGraph)
        self.graph = nx.Graph()
        
    @rule(target=st.builds(str), name=st.text(min_size=1))
    def add_node(self, node_name):
        self.graph.add_node(node_name)
        return f'add_node {node_name}'

    @rule(target=st.builds(str), source=st.sampled_from(self.graph.nodes), target=st.sampled_from(self.graph.nodes))
    def add_edge(self, sourceNode, targetNode):
        self.graph.add_edge(sourceNode, targetNode)
        return f'add_edge {sourceNode} {targetNode}'

    def teardown(self):
        transform_graph(self.graph)

graph_machine_steps = given(st.lists(GRAPH_ACTION_STRATEGY, min_size=1))

@graph_machine_steps
@settings(max_examples=50)
def test_graph_manipulation(steps):
    """
    Test graph transformations list and ensure they are able to be replayed in our vector machine
    """
    machine = GraphMachine()
    for step in steps:
        machine.execute_step(step)

# Define a Lark grammar
graph_grammar = """
    start: command+
    command: "add_node" CNAME
           | "add_edge" CNAME CNAME 
    CNAME: /[a-z][a-zA-Z0-9_]*/
    %import common.WS
    %ignore WS
"""

class TreeToGraph(Transformer):
    """
    class to convert parse tree to graph
    """
    def __init__(self):
        self.graph = nx.Graph()

    def start(self, commands):
        return self.graph

    def add_node(self, name):
        self.graph.add_node(str(name[0]))

    def add_edge(self, names):
        self.graph.add_edge(str(names[0]), str(names[1]))
        
    def CNAME(self, name):
        return str(name[0])

def transform_graph(graph: nx.Graph) -> nx.Graph:
    """Here transformation of graph could be implemented"""
    return graph


if __name__ == "__main__":
    test_graph_manipulation()