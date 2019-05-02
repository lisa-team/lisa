from .context import lisa
from lisa import matching, graph, mire_check
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


def test_match_single(kd):
    coord = (-77.01, 38.9)
    assert matching.match_single(coord, kd, t=0.001) == '11042442_12042442_15065322'


def test_match_trace(kd, G):
    trace = [(-77.0077, 38.915), (-77.0078, 38.9148), (-77.0081, 38.914)]
    assert matching.match_trace(trace, kd, G) == [
        '12055642_47046662', '12055642_12078912', '12055642_12077672', '12055642_12074622']


def test_get_closest_osmnx_path(kd, G):
    trace = [(-77.0077,  38.915), (-77.0078, 38.9148), (-77.0081, 38.914)]
    assert matching.get_closest_osmnx_path(trace, kd, G) == [
        '12055642_47046662', '12055642_12078912', '12055642_12074622']


def test_connect_path(G):
    path = ['12055642_47046662', '12055642_47046662', '12055642_12074622']
    assert matching.connect_path(path, G) == [
        '12055642_47046662', '12055642_12078912', '12055642_12077672', '12055642_12074622']


def test_connect_path_fails(G):
    raw_path = [(-77.01, 38.9), (-77.015, 38.85)]
    with pytest.raises(Exception):
        matching.connect_path(raw_path, G)


def test_make_best_guess(G):
    assert matching.make_best_guess('12055642_47046662', '12055642_12074622', G.init_graph) == [
        '12055642_47046662', '12055642_12078912', '12055642_12077672', '12055642_12074622']


def test_make_best_guess_fails(G):
    with pytest.raises(Exception):
        matching.make_best_guess(30, 31, G.init_graph)
