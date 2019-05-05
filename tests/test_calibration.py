from .context import lisa
from lisa import calibrate, closest_node, graph, mire_check
import pytest


@pytest.fixture
def G():
    '''
    Returns osmnx graph of DC
    '''
    return mire_check.get_expanded_graph_from_mire(
        '../lisa/scratch_022819.gdb', 3, 2)


@pytest.fixture
def kd(G):
    '''
    Returns kd tree
    '''
    return graph.KDTreeWrapper(G.init_graph)

