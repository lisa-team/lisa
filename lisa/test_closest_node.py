from closest_node import generate_line, calculate_dist, nearest_node
import pytest
from graph import KDTreeWrapper
import pickle
# python -m pytest test_closest_node.py -v


@pytest.fixture
def G():
    '''
    Returns osmnx graph of DC
    '''
    pickle_in = open('dc.pickle', "rb")
    return pickle.load(pickle_in)


@pytest.fixture
def kd(G):
    '''
    Returns kd tree
    '''
    return KDTreeWrapper(G.DiGraph)


@pytest.mark.parametrize("P, Q, res", [
    ({'x': 0, 'y': 1}, {'x': 4, 'y': 0}, (1, 4, -4)),
    ({'x': 2, 'y': 1}, {'x': 3, 'y': 0}, (1, 1, -3)),
    ({'x': 2, 'y': 1}, {'x': 5, 'y': 0}, (1, 3, -5)),
    ({'x': 1, 'y': 1}, {'x': 4, 'y': 0}, (1, 3, -4)),
    ({'x': 0, 'y': 1}, {'x': 2, 'y': 4}, (-3, 2, -2)),
])
def test_generate_line(P, Q, res):
    assert generate_line(P, Q) == res


@pytest.mark.parametrize("coord, a, b, c, res", [
    ((0, 0), 3, 4, -6, 1.2),
    ((-3, 7), 6, -5, 10, 5.506),
    ((5, 6), -2, 3, 4, 3.328)
])
def test_calculate_dist(coord, a, b, c, res):
    assert round(calculate_dist(coord, a, b, c), 3) == res


@pytest.mark.parametrize("coord", [
    (38.91, 77.042),
    (38.913, 77.039),
    (38.915, 77.045),
    (38.919, 77.031),
])
def test_nearest_node(coord, G, kd):
    node, dist = nearest_node(coord, kd, G.DiGraph)
    assert dist < 100
