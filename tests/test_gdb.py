from .context import lisa
from lisa import *
import os
import pytest
import matplotlib.pyplot as plt
from data import add_data_from_gdb
from graph import Graph, Name

# @pytest.fixture
def g():
    '''3
    Returns osmnx graph of DC
    '''
    filename = "WashingtonDCGraph.pkl"

    if not os.path.exists(filename):
        name = Name("Washington DC")
        dc = Graph(name)
        dc.save(filename)
    else:
        dc = Graph.from_file(filename)

    return dc

def print_nodes(g):
    print(g.DiGraph.nodes)
    assert 1 == 1

def verify_node_attirbutes(g):
    before = set()
    for n1, n2, data in g.DiGraph.edges(data=True):
        before.union(data.keys())

    add_data_from_gdb(g, "scratch_022819.gdb")
    after = set()
    for n1, n2, data in g.DiGraph.edges(data=True):
        after.union(data.keys())

    assert before != after

if __name__ == "__main__":
    verify_node_attirbutes(g())
