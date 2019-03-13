from collections import defaultdict
import numpy as np
import networkx as nx
from typing import List, Dict, Union, Iterable, Tuple, NewType
from nx_types import EdgesExpd, NodeExpd, NodesExpd, NodeExpdToXYNP


def xy_vec(nodes_xy: NodeExpdToXYNP, n1: NodeExpd, n2: NodeExpd) -> np.ndarray:
    """Calculates the vector that takes you from n1 to n2
    
    Arguments:
        nodes_xy {NodeExpdToXYNP} -- [description]
        n1 {NodeExpd} -- [description]
        n2 {NodeExpd} -- [description]
    
    Returns:
        np.ndarray -- [description]
    """
    return nodes_xy[n2] - nodes_xy[n1]


def segment_vec(nodes_xy: NodeExpdToXYNP, segment: EdgesExpd) -> np.ndarray:
    """Calculates the vector between two segments
    
    Args:
        nodes_xy (NodeExpdToXYNP): [description]
        segment (EdgesExpd): [description]
    
    Returns:
        np.ndarray: [description]
    """
    return xy_vec(nodes_xy, segment[0], segment[1])


def segment_unit_vec(nodes_xy: NodeExpdToXYNP, segment: EdgesExpd) -> np.ndarray:
    """[summary]
    
    Arguments:
        nodes_xy {NodeExpdToXYNP} -- [description]
        segment {EdgesExpd} -- [description]
    
    Returns:
        np.ndarray -- [description]
    """
    arr = segment_vec(nodes_xy, segment)
    return arr / np.linalg.norm(arr)


def create_nodes_xy(g: nx.MultiDiGraph, nodes: NodesExpd) -> Dict[NodeExpd, np.ndarray]:
    """[summary]
    
    Args:
        g (nx.MultiDiGraph): [description]
        nodes (NodesExpd): [description]
    
    Returns:
        Dict[NodeExpd, np.ndarray]: [description]
    """
    nodes_xy = {}
    for node in nodes:
        xy = g.nodes()[node[0]]  # extract the xy from the original node pre-expansion
        x = xy["x"]
        y = xy["y"]
        nodes_xy[node] = np.array((x, y))

    return nodes_xy


def update_nodes_xy(
    nodes_xy: Dict[NodeExpd, np.ndarray], segments: EdgesExpd
) -> Dict[NodeExpd, np.ndarray]:
    """[summary]

    Args:
        nodes_xy (Dict[NodeExpd, np.ndarray]): [description]
        segments (EdgesExpd): [description]

    Returns:
        Dict[NodeExpd, np.ndarray]: [description]

    TODO: Allow user to specify distance (width of lane)
    """
    dist = 0.00001  # change to 10 feet once you figure out units
    for segment in segments:
        if segment[2]["has_comp"]:
            unit_vec = segment_unit_vec(nodes_xy, segment)
            perp_vec = np.array([unit_vec[1], unit_vec[0] * -1])
            nodes_xy[segment[0]] = nodes_xy[segment[0]] + (
                (unit_vec * dist) + (perp_vec * dist / 2)
            )
            nodes_xy[segment[1]] = nodes_xy[segment[1]] - (
                (unit_vec * dist) - (perp_vec * dist / 2)
            )
        else:
            nodes_xy[segment[0]] = nodes_xy[segment[0]] + (unit_vec * dist)
            nodes_xy[segment[1]] = nodes_xy[segment[1]] - (unit_vec * dist)

    return nodes_xy


def create_nodes(nodes_xy_shifted: Dict[NodeExpd, np.ndarray]) -> NodesExpd:
    """[summary]
    
    Args:
        nodes_xy_shifted (Dict[NodeExpd, np.ndarray]): [description]
    
    Returns:
        NodesExpd: [description]
    """
    nodes_graph = {}
    for k, v in nodes_xy_shifted.items():
        nodes_graph[k] = {"x": v[0], "y": v[1]}
    nodes = [(k, v) for k, v in nodes_graph.items()]
    return nodes


def shift_nodes(g: nx.MultiDiGraph, nodes: NodesExpd, segments: EdgesExpd) -> NodesExpd:
    """Slightly shifts the x, y positions of expanded nodes 
    
    Args:
        g (nx.MultiDiGraph): [description]
        nodes (NodesExpd): [description]
        segments (EdgesExpd): [description]
    
    Returns:
        NodesExpd: [description]
    """
    nodes_xy = create_nodes_xy(g, nodes)
    nodes_xy_shifted = update_nodes_xy(nodes_xy, segments)
    nodes = create_nodes(nodes_xy_shifted)
    return nodes
