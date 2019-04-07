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

# For custom legends
import matplotlib as mpl
from matplotlib.lines import Line2D


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
            return (self._sorted_node_ids[idxs], distances)
        elif k > 1:
            return ([self._sorted_node_ids[i] for i in idxs], distances)


class GraphBuilder(object):
    """[summary]

    Raises:
        ValueError: [description]

    Returns:
        [type]: [description]
    """

    def __init__(self, bound: Bound, mire_graph: nx.Graph = None):
        """The "run" function to make Graph objects

        Args:
            bound (Union[Name, Bbox]): user desired bounds of the graph

        TODO: Make into callable function that returns a Graph object
        TODO: Figure out what should be saved as an attribute and what should be temp
        """
        # initialize graph from given bounx
        self.bound = bound
        if not mire_graph and self.bound:
            self.g_init = self.initialize_map(self.bound)
        else:
            self.g_init = mire_graph 

        # expand graph, add data (currently random)
        g_expd = expand_graph(self.g_init)
        if not mire_graph:
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
        int_edges = [(node_to_int[n1], node_to_int[n2], data)
                     for n1, n2, data in g.edges(data=True)]

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

    def __init__(self, bound: Union[Name, Bbox], mire_graph: nx.Graph = None):
        """[summary]

        Args:
            bound (Union[Name, Bbox]): [description]
        """
        if bound:
            graph_builder = GraphBuilder(bound)
        else:
            graph_builder = GraphBuilder(bound = None, mire_graph = mire_graph)
        self.DiGraph = graph_builder.g_expd_int
        self.init_graph = graph_builder.g_init
        self.node_map = graph_builder.init_to_expd_int_nodes
        self.init_min_dist = graph_builder.init_kd_tree
        self.dg_min_dist = graph_builder.dg_kd_tree

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

    def plot_simple_graph(self, fig_height=10):
        ox.plot_graph(self.init_graph, fig_height=fig_height)

    def plot_routes(self, routes, fig_height=10):
        """
        Create_mdg() appears to be nondeterministic.
        routes is a list of routes.
            Each route is a list of nodes traversed in order.

        routes = None picks two routes of length 1 and plots those.


        """
        MDG = self.create_mdg()
        if routes:
            ox.plot_graph_routes(MDG, routes, fig_height=fig_height)
        else:
            first_node_list = [list(MDG.edges)[0][0], list(MDG.edges)[0][1]]
            second_node_list = [list(MDG.edges)[1][0], list(MDG.edges)[1][1]]

            routes = [first_node_list, second_node_list]
            ox.plot_graph_routes(MDG, routes, fig_height=fig_height)

    def create_legend(self, edge_legend, node_legend):
        legend_elements = []
        if edge_legend:
            for edge_label in edge_legend:
                legend_elements.append(
                    Line2D([0], [0], color=edge_legend[edge_label], lw=3, label=edge_label))

        if node_legend:
            for node_label in node_legend:
                legend_elements.append(Line2D([0], [0], marker='o', color=node_legend[node_label], label=node_label,
                                              markerfacecolor=node_legend[node_label], markersize=8))
        return legend_elements

    def highlight_graph(self, edge_filter_function, node_filter_function, legend_elements, title):
        """
        edge_filter_function and node_filter_function take in a dict and return a color.

        something like:

        edge_filter_function = lambda x: 'r' if x.get("traffic_volume",0)>200 else '#0F0F0F'

        node_filter_function = lambda z: 'b' if z.get("y")>-77.098 else '#0F0F0F'

        """

        G = self.create_mdg()
        ec = '#0F0F0F'
        nc = '#0F0F0F'

        if edge_filter_function:
            ec = [edge_filter_function(data) for u, v, data in G.edges(data=True)]
        if node_filter_function:
            nc = [node_filter_function(data) for u, data in G.nodes(data=True)]

        fig, ax = ox.plot.plot_graph(G, show=False, close=False, edge_color=ec, node_color=nc)

        ax.set_title(title)
        ax.legend(handles=legend_elements)

        return fig, ax

    def show_graph(self):
        plt.show()

    def create_pos(self):
        pos = {}
        for tup in self.DiGraph.nodes(data=True):
            node = tup[0]
            xy_dict = tup[1]
            pos[node] = (xy_dict["x"], xy_dict["y"])
        return pos

    def create_reverse_pos(self):
        pos = {}
        for tup in self.DiGraph.nodes(data=True):
            node = tup[0]
            xy_dict = tup[1]
            pos[(xy_dict["x"], xy_dict["y"])] = node
        return pos

    def create_edge_labels(self, attribute_list):
        edge_labels = {(u, v): {attribute: d.get(attribute) for attribute in attribute_list}
                       for u, v, d in self.DiGraph.edges(data=True)}
        return edge_labels


class Graph_Hover(object):
    """Object for displaying graph edge attributes on hover. One object must be created for each axis.
    """

    def __init__(self, graph, fig, ax, node_hover=False):
        """Initializes the Graph_Hover object.

        Args:
            graph (lisagraph.Graph): A Graph object to display.
            fig (matplotlib.figure.Figure): A Figure object. Usually returned by calling graph's highlight_graph() method.
            ax (matplotlib.axes.Axes): An Axes object. Usually returned by calling graph's highlight_graph() method
        Returns:
            Graph_Hover object.
        """
        self.graph = graph
        self.fig = fig
        self.ax = ax
        self.reverse_pos = graph.create_reverse_pos()
        self.annot = None

        # For node hovering
        self.node_hover = node_hover
        if self.node_hover:
            self.mdg = self.graph.create_mdg()

    def annotate_ax(self):
        """Creates the annotation textbox.
        """
        self.annot = self.ax.annotate("", xy=(0, 0), xytext=(-20, 20), textcoords="offset points",
                                      bbox=dict(boxstyle="round", fc="w"),
                                      arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)

    def hover(self, event):
        """Detects mouse movement and displays annotation when mouse is over graph edges.
        """
        axis = event.inaxes

        if axis == self.ax:
            annot = self.annot
            vis = self.annot.get_visible()
            for child in axis.collections:
                if type(child) is mpl.collections.LineCollection:
                    cont, ind = child.contains(event)
                    if cont:

                        arr = ind["ind"]

                        annot.xy = (event.xdata, event.ydata)

                        # point1 and point2 are each [x, y] arrays
                        point1, point2 = child.get_segments()[arr[0]]
                        point1_x, point1_y = point1
                        point2_x, point2_y = point2

                        node1 = self.reverse_pos[(point1_x, point1_y)]
                        node2 = self.reverse_pos[(point2_x, point2_y)]
                        edge = self.graph.DiGraph.edges[node1, node2]

                        text = str(edge)
                        self.annot.set_text(text)
                        self.annot.get_bbox_patch().set_alpha(0.4)

                        self.annot.set_visible(True)
                        fig.canvas.draw_idle()
                    else:
                        if vis:
                            self.annot.set_visible(False)
                            fig.canvas.draw_idle()

                # For node hovering
                elif self.node_hover and type(child) is mpl.collections.PathCollection:
                    cont, ind = child.contains(event)
                    if cont:
                        arr = ind["ind"]
                        # print(arr[0])
                        # super sketchy, but apparently self.mdg.nodes names the nodes the same way pyplot reads them
                        text = self.mdg.nodes(data=True)[arr[0]]

                        self.annot.set_text(text)
                        self.annot.get_bbox_patch().set_alpha(0.4)

                        self.annot.set_visible(True)
                        fig.canvas.draw_idle()
                    else:
                        if vis:
                            self.annot.set_visible(False)
                            fig.canvas.draw_idle()

    def add_scatter(self, x_s: list, y_s: list, color: str):
        """Add scatterplot points on top of the displayed axis.
        Args:
            x_s (list): A list of the x coordinates of each point to be added.
            y_s (list): A list of the y coordinates of each point to be added.
            color (str): A string representing the color of all points to be added.
        Returns:
            None
        """
        self.ax.scatter(x_s, y_s, color=color)

    def display_graph(self):
        """Connects self.hover to the figure and displays the graph.
        """
        self.annotate_ax()
        self.fig.canvas.mpl_connect("motion_notify_event", self.hover)
        plt.show()

    def save_fig(self, filename):
        plt.savefig(filename+'.png')

if __name__ == "__main__":


    test = nx.DiGraph()
    test.add_nodes_from([1,2,3])
    test.add_edges_from([(1,2, {1:2}), (2,3), (3,1)])

    # print(test.nodes())
    # print(test.edges())

    # print(test.edges()[1,2])

    print([k for k in nx.strongly_connected_components(test)])




    bbox = Bbox(38.883_000_16, 38.878_726_840_000_006, -77.099_398_32, -77.105_007_68)
    # bbox = Name("Washington, DC")
    # bbox = Bbox(38.898191, 38.894810, -77.003528, -77.010062)
    G = Graph(bbox)
    print(f"First 100 nodes: {list(G.DiGraph.nodes)[:100]}\n")
    print(f"First 100 edges: {list(G.DiGraph.edges)[:100]}\n")

    # init_graph_node = list(G.init_graph.nodes)[30]  # pink node
    # expanded_nodes = G.node_map[init_graph_node]  # yellow nodes
    # print(f"Pink node: {init_graph_node} -> Yellow nodes: {expanded_nodes}\n")

    def edge_filter(data):
        if data.get("separate_path"):
            return 'r'
        elif data.get("crosswalk"):
            return 'm'
        elif data.get("bike_lane"):
            return 'g'
        else:
            return "#1F1F1F"

    def node_filter(data):
        if data.get("x") > -77.101:
            return 'b'
        else:
            return "#1F1F1F"

    edge_legend = {"Separate path": 'r', "Has crosswalk": 'm', "Has bike lane": 'g'}
    node_legend = {"x > -77.01": 'b', "x <= -77.01": '#1F1F1F'}

    edge_and_nodes = G.create_legend(edge_legend=edge_legend, node_legend=node_legend)
    only_nodes = G.create_legend(edge_legend=None, node_legend=node_legend)

    # fig, ax1 = G.highlight_graph(edge_filter_function=edge_filter,
    #                              node_filter_function=node_filter, legend_elements=edge_and_nodes, title="Test title")


    # hover = Graph_Hover(graph=G, fig=fig, ax=ax1)
    # hover.save_fig("test")
    # hover.display_graph()

    # G.plot_simple_graph()
    # fig, ax2 = G.highlight_graph(edge_filter_function=None, node_filter_function=None, legend_elements=None, title = "")
    # ax2.add_scatter([-77.102, -77.103], [38.88, 38.881], 'b')


    # plt.show()

    # hover = Graph_Hover(graph=G, fig=fig, ax=ax2)
    # hover.add_scatter(x_s=[-77.102, -77.103], y_s=[38.88, 38.881], color='b')
    # hover.save_fig("E St and North Capitol St (1)")
    # hover.display_graph()
