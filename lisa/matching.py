""" Functions used for map matching

This module contains functions for matching long/lat coordinates to our graph
representations. There are two main functions here, one for matching a single
long/lat (for example a stop sign) to a node, and one for matching multiple
coordinates to a path through the graph.
"""


def match_single(coord, kd):
    """Match (long, lat) to closest osmnx graph node.
    Args:
        coord (Tuple[float, float]): the latitude and longitude to match
        kd: the kd tree built off init_graph
    Returns:
        int: the best-match node ID
    """
    closest_node, d = kd.query_min_dist_nodes(coord)
    if d > 0.00001:
        raise Exception('Closest node is not a match. d: ', d)
    return closest_node


def match_trace(trace, kd, G):
    """Match a list of (long,lat) coordinates to the most likely cycling route
    in the osmnx graph.
    Args:
        trace (List[Tuple[float, float]]): a list of (lat, long) points to
            match
        kd: the kd tree built off init_graph
    Returns:
        [int]: the list of best match node IDs
    """

    def get_closest_osmnx_path():
        """
        For each (long,lat) in map_box_path, return the closest node in the
        osmnx graph
        Args:
            map_box_path (List[Tuple[float, float]]): a list of (lat, long)
                coordinates
            kd: the kd tree built off init_graph
        Returns:
            (List[Tuple[float, float]]): the list of best match node IDs
        """
        path = []
        for coord in trace:
            closest_node, d = kd.query_min_dist_nodes(coord)
            if d > 0.00001:
                continue
            path.append(closest_node)
        return path

    def is_valid_path(osmnx_path):
        """
        Check if osmnx_path is valid.
        Args:
            osmnx_path (List[Tuple[float, float]]): a list of (lat, long)
                coordinates
        Returns:
            (List[Tuple[float, float]]): the input list
        """
        for i in range(0, len(osmnx_path)-1):
            curr_node = osmnx_path[i]
            next_node = osmnx_path[i+1]
            neighbors = G.neighbors(curr_node)
            if next_node not in neighbors:
                print('Nodes do not form valid path')
                return
        return osmnx_path

    osmnx_match = get_closest_osmnx_path()
    if is_valid_path(osmnx_match):
        return osmnx_match
