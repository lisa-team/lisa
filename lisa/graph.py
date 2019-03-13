from collections import defaultdict
from typing import Union, List, Dict
import pickle
import osmnx as ox
import numpy as np
import networkx as nx
from shapely.geometry import Polygon, MultiPolygon
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from expansion import create_dg, expand_graph, create_node_map
from data import add_random_attributes


class Name(object):
    """[summary]
    
    Raises:
        TypeError: [description]

    TODO: Finish docstrings
    """

    def __init__(self, name: str):
        """[summary]
        
        Args:
            name (str): [description]
        
        Raises:
            TypeError: [description]
        """
        self.name = name

        self.gdf = ox.gdf_from_place(name)
        self.official_name = self.gdf.place_name.values[0]
        self.geometry = self.gdf.geometry[0]
        if not type(self.geometry) is Polygon or type(self.geometry) is MultiPolygon:
            raise TypeError("Location geometry was not a Polygon or a MultiPolygon")

    def summary(self):
        """[summary]
        """
        print(f"Input Name: {self.name}")
        print(f"Official Name: {self.official_name}")
        print(type(self.geometry))


class Bbox(object):
    """[summary]
    
    Returns:
        [type]: [description]

    TODO: Refactor into namedtuple
    """

    def __init__(self, north, south, east, west):
        """[summary]
        
        Args:
            north ([type]): [description]
            south ([type]): [description]
            east ([type]): [description]
            west ([type]): [description]
        """
        self.bbox = (north, south, east, west)

    def __iter__(self):
        """[summary]
        
        Returns:
            [type]: [description]
        """
        return (b for b in self.bbox)

    def summary(self):
        """[summary]
        """
        width = self.bbox[0] - self.bbox[1]
        height = self.bbox[2] - self.bbox[3]
        area = width * height
        print(f"Width: {width}")
        print(f"Height: {height}")
        print(f"Area: {area}")

Bound = Union[Name, Bbox]

class KDTreeWrapper(object):
    """Wraps the Scipy KD Tree to allow minimum distance node querying of NetworkX graphs.
    """

    def __init__(self, g):
        """Initializes a KD tree and corresponding sorted node arrays
        
        Args:
            g (nx.DiGraph): input DiGraph
        """
        self._sorted_nodes = sorted(list(g.nodes(data=True)), key=lambda n: n[0])
        self._sorted_node_ids = [node for node, data in self._sorted_nodes]
        self._xys = [(data["x"], data["y"]) for node, data in self._sorted_nodes]
        self._kd = KDTree(self._xys)

    def query_min_dist_nodes(self, x, k=1, distance_upper_bound=np.inf):
        """Finds the closest "k" nodes to the point "x" with maximum distance "distance_upper_bound"
        
        Args:
            x : point with len 2 for x and y coordinates
            k (int, optional): Defaults to 1. number of closest nodes to find
            distance_upper_bound ([type], optional): Defaults to np.inf. maximum distance for found node
        
        Returns:
            Union[Node, List[Node]]: closest k nodes

        TODO: Fix type of x
        """
        distances, idxs = self._kd.query(
            x, k=k, distance_upper_bound=distance_upper_bound
        )
        if k == 1:
            return self._sorted_node_ids[idxs]
        elif k > 1:
            return [self._sorted_node_ids[i] for i in idxs]


class GraphBuilder(object):
    """[summary]
    
    Raises:
        ValueError: [description]
    
    Returns:
        [type]: [description]
    """

    def __init__(self, bound: Bound):
        """The "run" function to make Graph objects
        
        Args:
            bound (Union[Name, Bbox]): user desired bounds of the graph

        TODO: Make into callable function that returns a Graph object
        TODO: Figure out what should be saved as an attribute and what should be temp
        """
        # initialize graph from given bounx
        self.bound = bound
        self.g_init = self.initialize_map(self.bound)

        # expand graph, add data (currently random)
        g_expd = expand_graph(self.g_init)
        g_expd = add_random_attributes(g_expd)
        init_to_expd_nodes = create_node_map(self.g_init, g_expd)

        # convert expanded graph's nodes to integers
        self.g_expd_int = self.enumerate_graph(g_expd)
        self.init_to_expd_int_nodes = self.enumerate_node_mapping(g_expd, init_to_expd_nodes)

        # add auxiliary data structures
        self.init_kd_tree = KDTreeWrapper(self.g_init)
        self.dg_kd_tree = KDTreeWrapper(self.g_expd_int)

    def initialize_map(self, bound: Union[Name, Bbox]):
        """initialize_map takes in a bound and uses osmnx to create an inital map of the desired area.
        
        Args:
            bound (Union[Name, Bbox]): user desired bounds of the graph 
        
        Raises:
            ValueError: invalid bound type
        
        Returns:
            [type]: [description]
        """
        init_graph = None
        if type(bound) is Name:
            init_graph = ox.graph_from_place(bound.official_name)
        elif type(bound) is Bbox:
            init_graph = ox.graph_from_bbox(*bound)
        else:
            raise ValueError("Could not create graph from specified bound")
        return init_graph

    def node_to_int(self, g: nx.DiGraph):
        """Creates a dictionary mapping all the nodes in a graph to integers
        
        Args:
            g (nx.DiGraph): [description]
        
        Returns:
            [type]: [description]
        """
        return {node: i for i, node in enumerate(g.nodes)}

    def enumerate_node_mapping(self, g: nx.DiGraph, node_map: dict):
        """[summary]
        
        Args:
            g (nx.DiGraph): [description]
            node_map (dict): [description]
        
        Returns:
            [type]: [description]
        """
        node_to_int = self.node_to_int(g)
        return {k: [node_to_int[n] for n in ns] for k, ns in node_map.items()}

    def enumerate_graph(self, g: nx.DiGraph):
        """Creates a graph with any node type from a graph with integer nodes
        
        Args:
            g (nx.DiGraph): [description]
        
        Returns:
            [type]: [description]
        """
        node_to_int = self.node_to_int(g)

        int_nodes = [(node_to_int[node], data) for node, data in g.nodes(data=True)]
        int_edges = [(node_to_int[n1], node_to_int[n2], data) for n1, n2, data in g.edges(data=True)]

        return create_dg(int_nodes, int_edges)

    def plot_map(self, fig_height: int = 10):
        """Helper function to the initial

        Args:
            fig_height (int, optional): Defaults to 10. [description]
        """
        ox.plot_graph(self.g_init, fig_height=fig_height)


class Graph(object):
    """
    A wrapper for nxgraphs that we should probably have

    Allows a one way conversion from DiGrahs to MultiDiGraphs to utilize built in
    osmnx visualization features
    """

    def __init__(self, bound: Union[Name, Bbox]):
        """[summary]
        
        Args:
            bound (Union[Name, Bbox]): [description]
        """
        graph_builder = GraphBuilder(bound)
        self.DiGraph = graph_builder.g_expd_int
        self.init_graph = graph_builder.g_init
        self.node_map = graph_builder.init_to_expd_int_nodes
        self._init_min_dist = graph_builder.init_kd_tree
        self._dg_min_dist = graph_builder.dg_kd_tree

    @staticmethod
    def from_file(filepath: str):
        """Unpickle a graph object
        
        Args:
            filepath (str): filepath to the object
        
        Returns:
            Graph: [description]
        """
        with open(filepath, "rb") as f:
            return pickle.load(f)

    def save(self, filepath: str):
        """Pickle graph object to a file
        
        Args:
            filepath (str): filepath to save the object
        """
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    def create_mdg(self):
        """Create a MultiDiGraph from self.DiGraph for visualization purposes
        
        Returns:
            ox.MultiDiGraph -- []
        """
        G = nx.MultiDiGraph()

        # hard coding graph parameters for visualization purposes
        G.graph = {
            "name": "Visualization Graph",
            "crs": {"init": "epsg:4326"},
            "simplified": True,
        }
        G.add_nodes_from(self.DiGraph.nodes(data=True))
        G.add_edges_from(
            [(n1, n2, 0, data) for n1, n2, data in self.DiGraph.edges(data=True)]
        )
        return G

    def plot_graph(self, fig_height=10):
        MDG = self.create_mdg()
        ox.plot_graph(MDG, fig_height=fig_height)


if __name__ == "__main__":
    bbox = Bbox(38.883_000_16, 38.878_726_840_000_006, -77.099_398_32, -77.105_007_68)
    G = Graph(bbox)
    print(f"First 100 nodes: {list(G.DiGraph.nodes)[:100]}\n")
    print(f"First 100 edges: {list(G.DiGraph.edges)[:100]}\n")

    init_graph_node = list(G.init_graph.nodes)[30]  # pink node
    expanded_nodes = G.node_map[init_graph_node]  # yellow nodes
    print(f"Pink node: {init_graph_node} -> Yellow nodes: {expanded_nodes}\n")
