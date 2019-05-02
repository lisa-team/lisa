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
                    filemode='w+',
                    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')


class CoordinateMatchError(Exception):
    """Exception for unfound coordinate/graph match"""
    pass


def match_single(coord, kd, t=0.0001):
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
    in the nx graph

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
        if G.is_valid_path(match):
            return match
    except Exception as e:
        logging.info(e)


def match_paths(paths, kd, G):
    """match a list of gps paths to their nx node ids

    Args:
        paths(list): list of list of long/lat coordinates
        kd(): kdtree object
        G: nx graph object
    Returns:
        list of list of osmnx node ids
    """
    matched_paths = []
    for raw_path in paths:
        match = match_trace(raw_path, kd, G)
        matched_paths.append(match)

    expd_paths, failed_paths = G.init_paths_to_expd(
        list(filter(None, matched_paths)), False)
    if failed_paths:
        raise Exception('Found ' + str(len(failed_paths)) +
                        ' invalid path(s) during graph expansion')

    return expd_paths


"""
Helper functions
"""


def get_closest_osmnx_path(trace, kd, G):
    """Get the closest node for each coord in trace

    Args:
        trace: a list of (lat, long) coordinates
        kd: KDTreeWrapper object
        G: Graph object
    Returns:print
        path(list): the list of nx node IDs
    """
    path = []
    for coord in trace:
        closest_node, d = nearest_node(coord, kd, G.init_graph)
        if d > 0.001:
            continue
        path.append(closest_node)
    if len(path) < 1:
        raise Exception('All nodes missed')
    return path


def connect_path(nx_raw_path, G):
    """Convert nx_raw_path to connected nx path.

    Args:
        nx_raw_path: a list of nx node ids
        G: Graph object
    Returns:
        list of node ids
    """
    path = [nx_raw_path[0]]
    for i in range(0, len(nx_raw_path)-1):
        curr = nx_raw_path[i]
        nxt = nx_raw_path[i+1]
        if G.are_neighbors(curr, nxt):
            path.append(nxt)
        else:
            path += make_best_guess(curr, nxt, G.init_graph)[1:]
    if len(path) > len(nx_raw_path)*2:
        raise Exception('Path is too long to constitute valid route')
    if len(path) < 2:
        raise Exception('Path is too short to form a route')
    return path


def make_best_guess(base, target, G):
    """Return shortest path between two nodes in an nx graph.

    Args:
        base (int): nx node id
        target (int): nx node id
        G: MultiDiGraph object
    Returns:
        (List[int]): list of integer node ids
    """
    try:
        return nx.shortest_path(G, base, target)
    except nx.exception.NetworkXNoPath:
        raise Exception('Path cannot be connected')
