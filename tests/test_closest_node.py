from .context import lisa
from lisa import closest_node, graph, mire_check
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


@pytest.mark.parametrize("P, Q, res", [
    ({'x': 0, 'y': 1}, {'x': 4, 'y': 0}, (1, 4, -4)),
    ({'x': 2, 'y': 1}, {'x': 3, 'y': 0}, (1, 1, -3)),
    ({'x': 2, 'y': 1}, {'x': 5, 'y': 0}, (1, 3, -5)),
    ({'x': 1, 'y': 1}, {'x': 4, 'y': 0}, (1, 3, -4)),
    ({'x': 0, 'y': 1}, {'x': 2, 'y': 4}, (-3, 2, -2)),
])
def test_generate_line(P, Q, res):
    assert closest_node.generate_line(P, Q) == res


@pytest.mark.parametrize("coord, a, b, c, res", [
    ((0, 0), 3, 4, -6, 1.2),
    ((-3, 7), 6, -5, 10, 5.506),
    ((5, 6), -2, 3, 4, 3.328)
])
def test_calculate_dist(coord, a, b, c, res):
    assert round(closest_node.calculate_dist(coord, a, b, c), 3) == res
