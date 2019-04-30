from collections import defaultdict
import numpy as np
from shapely.geometry import MultiPolygon
import networkx as nx
from typing import List, Dict, Union, Iterable, Tuple, NewType
from node_geometry import shift_nodes
from typing import *
from nx_types import *

def create_in_out_dict(g: nx.MultiDiGraph):
    """Creates a dictionary where the keys (nodes in a graph) are associated with
    another dictionary that tells what nodes can be traversed to by following
    edges "out" of the key node and what edges lead "in" to the key node
    
    Args:
        g (nx.MultiDiGraph): input graph whos nodes and edges are used to create in_out dict
    
    Returns:
        dict: in out dictionary of node traversals

    TODO: make if statement clearer - remove single loops in graph
    """
    in_out = defaultdict(lambda: {"in": [], "out": []})
    for start, end in g.edges():
        if start == end:
            continue
        in_out[end]["in"].append(start)
        in_out[start]["out"].append(end)
    return in_out

def create_dg(nodes: NodesExpd, edges: EdgesExpd) -> nx.DiGraph:
    """
    Creates a networkx DiGraph given a list of nodes and a list of edges
    """
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    return G


def expand_graph(g_init: nx.MultiDiGraph):
    """ 
    Expands a networkx MultiDiGraph from a single node intersection representation
    to a multi node intersection representation.

    :param g_init: The graph to expand
    :type g_init: nx.MultiDiGraph

    :returns: nx.DiGraph
    """
    in_out = create_in_out_dict(g_init)
    segments = SegmentBuilder.create_segments(in_out, g_init)
    nodes = NodeBuilder.extract_nodes(segments)

    nodes = shift_nodes(g_init, nodes, segments)

    intersections = IntersectionBuilder.create_intersections(in_out, g_init)

    edges = segments + intersections
    g_expanded = create_dg(nodes, edges)
    return g_expanded


def create_node_map(g_initial: nx.MultiDiGraph, g_expanded: nx.DiGraph):
    """
    Creates a node map associating the intitial nodes with the corresponding
    expanded nodprint(node)es. This will allow us to add data to an entire intersection by
    just locating the closest node in the initial graph.
    """
    node_map = {x: [] for x in g_initial.nodes}
    for node in g_expanded.nodes:
        if node[2] == "in":
            node_map[node[0]].append(node)
        elif node[2] == "out":
            node_map[node[1]].append(node)
        else:
            raise Exception(f"Found bad node: {node}")
    return node_map



class IntersectionBuilder(object):
    @staticmethod
    def create_intersections(in_out, g: nx.MultiDiGraph):
        """
        Expands nodes to intersection edges - keeping all node data in each edge

        Args:
            g (nx.MultiDiGraph): input graph used to pull node data
            in_out (Dict): in out dictionary of the init_graph nodes
        """
        edgeID = 0
        intersections = []
        for k, v in in_out.items():
            # every in connects to every out - unless same node
            for n_in in v["in"]:
                n1 = (k, n_in, "in")
                for n_out in v["out"]:
                    n2 = (k, n_out, "out")
                    if n_in != n_out:
                        edge_data = {}

                        edge_data.update(g.nodes(data=True)[k])
                        edge_data["type"] = "intersection"


                        edge_data["edgeID"] = edgeID
                        edgeID+=1


                        intersections.append((n1, n2, edge_data))

        return intersections


class NodeBuilder(object):
    @staticmethod
    def extract_nodes(segments):
        nodes = set()
        for segment in segments:
            nodes.add(segment[0])
            nodes.add(segment[1])
        return nodes


class SegmentBuilder(object):
    def complement_dir(self, s: str):
        """
        TODO: Switch to True and False so I don't have to write this function
        """
        if s == "in":
            return "out"
        elif s == "out":
            return "in"
        else:
            print("complement_dir failed")

    def complement_segment(self, segment):
        """
        Computes the complementary segment of the given segment. The
        complementary segement represents the other direction in a two
        way street.
        """
        n0 = segment[0]
        n1 = segment[1]

        n3 = (n0[0], n0[1], self.complement_dir(n0[2]))
        n2 = (n1[0], n1[1], self.complement_dir(n1[2]))

        return (n2, n3)

    def create_segments_set(self, in_out):
        segments_set = set()
        for k, v in in_out.items():
            for node in v["in"]:
                n1 = (node, k, "out")
                n2 = (k, node, "in")
                segments_set.add((n1, n2))
        return segments_set

    def create_segments_list(self, segments, g: nx.MultiDiGraph):
        segment_list = []
        for segment in segments:
            data = g.edges[(segment[0][0], segment[0][1], 0)]
            data["type"] = "segment"
            data["has_comp"] = self.complement_segment(segment) in segments
            segment_list.append((segment[0], segment[1], data))
        return segment_list

    @classmethod
    def create_segments(cls, in_out, g: nx.MultiDiGraph):
        segments_set = cls().create_segments_set(in_out)
        segments_list = cls().create_segments_list(segments_set, g)
        return segments_list
