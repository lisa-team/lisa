""" Functions used for map matching

This module contains functions for matching long/lat coordinates to our graph
representations. There are two main functions here, one for matching a single
long/lat (for example a stop sign) to a node, and one for matching multiple
coordinates to a path through the graph.
"""
from closest_node import nearest_node
import networkx as nx
import logging

logging.basicConfig(level=logging.INFO,
                    filename='matching.log',
                    filemode='w',
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


class CoordinateMatchError(Exception):
    """Exception for unfound coordinate/graph match"""
    pass


def match_single(coord, kd, t=0.00001):
    """Match (long, lat) to closest osmnx graph node.
    Args:
        coord (Tuple[float, float]): the latitude and longitude to match
        kd (lisa.graph.KDWrapper): the kd tree built off init_graph
        t (float): tolerance in degrees
    Returns:
        int: the closest osmnx node ID
    """
    closest_node, d = kd.query_min_dist_nodes(coord)
    if d > t:
        raise CoordinateMatchError('Closest node is not a match. d: ', d)
    print("SUCCESSFUL MATCH closest node match d: ", d)
    return closest_node


def match_trace(trace, kd, G):
    """Match a list of (long,lat) coordinates to the best match cycling route
    in the osmnx graph

    Args:
        trace (List[Tuple[float, float]]): a list of (lat, long) points to
            match
        kd: KDTreeWrapper object
        G: Graph object
    Returns:
        list: the list of best match node IDs
    """
    try:
        osmnx_match = get_closest_osmnx_path(trace, kd, G)
        match = connect_path(osmnx_match, G)
        if is_valid_path(match, G):
            return match
    except Exception as e:
        logging.info(e)


def get_closest_osmnx_path(trace, kd, G):
    """Get the closest node for each coord in trace

    Args:
        trace: a list of (lat, long) coordinates
        kd: KDTreeWrapper object
        G: Graph object
    Returns:
        path(list): the list of osmnx node IDs
    """
    path = []
    for coord in trace:
        closest_node, d = nearest_node(coord, kd, G)
        if d > 0.00001:
            continue
        path.append(closest_node)
    if len(path) < 1:
        raise Exception('All nodes missed')
    return path


def is_valid_path(osmnx_path, G):
    """Check if osmnx_path is valid.

    Args:
        osmnx_path (List[Tuple[float, float]]): a list of (lat, long)
            coordinates
        G: Graph object
    Returns:
        boolean: True if node ids are connected to form valid osmnx path,
            otherwise False
    """
    for i in range(0, len(osmnx_path)-1):
        curr = osmnx_path[i]
        nxt = osmnx_path[i+1]
        if not are_neighbors(curr, nxt, G):
            return False
    return True


def connect_path(raw_path, G):
    """Convert raw_path to connected osmnx path.

    Args:
        raw_path (List[Tuple[float, float]]): a list of (lat, long)
            coordinates
        G: Graph object
    Returns:
        (List[Tuple[float, float]]): the input list
    """
    path = [raw_path[0]]
    for i in range(0, len(raw_path)-1):
        curr = raw_path[i]
        nxt = raw_path[i+1]
        if are_neighbors(curr, nxt, G):
            path.append(nxt)
        else:
            path += make_best_guess(curr, nxt, G)[1:]
    if len(path) > len(raw_path)*2:
        raise Exception('Path is too long to constitute valid route')
    if len(path) < 2:
        raise Exception('Path is too short to form a route')
    return path


def are_neighbors(base, target, G):
    """Check if target is a neighbor of base.

    Args:
        base (int): osmnx node id
        target (int): osmnx node id
        G: Graph object
    Returns:
        boolean
    """
    neighbors = set(G.neighbors(base))
    if target in neighbors:
        return True
    return False


def make_best_guess(base, target, G):
    """Return shortes path between two nodes in an osmnx graph.

    Args:
        base (int): osmnx node id
        target (int): osmnx node id
        G: Graph object
    Returns:
        (List[int]): list of integer node ids
    """
    try:
        return nx.shortest_path(G, base, target)
    except nx.exception.NetworkXNoPath:
        raise Exception('Path cannot be connected')
