# -*- coding: utf-8 -*-
""" A collection of functions used for data import/export and configuration.

This module contains the functions required for manipulating input data for use
in the rest of the package, and also export to standard formats.

Example:
    TODO insert some examples if needed

"""

# TODO some of what Nathan, Uma, and Annie have been writing goes here
import math

from matching import match_single, CoordinateMatchError
import networkx as nx
import fiona

from random import random, randrange, choice
from copy import deepcopy
from enum import Enum
from itertools import permutations


def shapefile_to_osmnx_graph(blah: str, blah2: str, blah3: int):
    """This is a single line describing function.

    This is your more detailed breakdown of the functions inner workings if
    necessary. This can be multiple lines, and is separated from the one liner
    with a blank line.

    Args:
        blah (str): a word and nothing more
        blah2 (str): is blah even a word?
        blah3 (int): blah3 is not a word

    Returns:
        bool: The return value. True for success, False otherwise.

    """
    pass


class IntersectionGrade(Enum):
    """Enumerates intersection grade codes

    reference: https://wiki.ddot.dc.gov/display/GIS/DDOT%27s+Transportation+Data+Products
    """
    AtGrade = 0
    NotAtGrade = 1
    Undefined = 2
    Uncontrolled = 10
    TwoWayStop = 11
    AllWayStop = 12
    SignalizedWithPedSignal = 13
    SignalizedWithNoPedSignal = 14
    SignalizedRRCrossing = 15
    SignedRRCrossing = 16
    YieldSign = 17


def add_data_from_gdb(g, filename: str):
    """Tags the input graph with data from the supplied geodatabase

    For each feature in the "BlockIntersection" layer of the geodatabase the
    closest intersection is found in G and all intersection edges are tagged
    with the Intersection Grade Code for that intersection.

    This program also assumes coordinates in seconds, not degrees

    reference: https://wiki.ddot.dc.gov/display/GIS/DDOT%27s+Transportation+Data+Products

    Args:
        g (lisa.graph.Graph): the graph to be tagged with intersection attributes
        filename (str): a relative filepath to the source geodatabas (.gbd file)

    Returns:
        None
    """

    layers_list = fiona.listlayers(filename)
    try:
        block_intersection_layer_i = layers_list.index("BlockIntersection")
    except ValueError as e:
        print("Supplied geodatabase file has no 'BlockIntersection' layer")
        raise e

    with fiona.open(filename, 'r', layer=block_intersection_layer_i) as inp:
        max_x = -math.inf
        min_x = math.inf
        max_y = -math.inf
        min_y = math.inf

        for f in inp:
            coords = f['geometry']['coordinates']
            coords = coords[0]/(60*60*2), coords[1]/(60*60)
            max_x = max(max_x, coords[0])
            min_x = min(min_x, coords[0])
            max_y = max(max_y, coords[1])
            min_y = min(min_y, coords[1])

            gc = f['properties']['GRADE']  # grade code

            if gc not in [0, 1, 2, 10, 11, 12, 13, 14, 15, 16, 17]:
                # raise ValueError("Unknown intersection grade code {}".format(gc))
                print("Unknown intersection grade code {}".format(gc))
                continue

            attrs = {
                'grade': gc,
                'notAtGrade': 1 if gc == 1 else 0,
                'stops': 4 if gc == 12 else 2 if gc == 11 else 0,
                'signal': 1 if gc in [13, 14] else 0,
                'pedsignal': 1 if gc == 14 else 0,
                'rr': 1 if gc == 15 else 0,
                'yield': 1 if gc == 17 else 0,
            }

            try:
                intersection = match_single(coords, g.init_min_dist)
            except CoordinateMatchError as e:
                print(e)
                continue
            nx.set_node_attributes(g.init_graph, {intersection: attrs})

            # set value for each node in expanded graph
            nodes = g.node_map[intersection]
            for p in permutations(nodes, 2):
                if g.DiGraph.has_edge(p):
                    g.DiGraph.set_edge_attributes({p: attrs})
        print("x range: [{}, {}]".format(min_x, max_x))
        print("y range: [{}, {}]".format(min_y, max_y))


class StreetDataGenerator:
    def random_intersection(self):
        """Creates random attributes for a intersection edge in a DiGraph.
        
        random_intersection does not check for already existing attributes in the edge and will subsequently
        overwrite any data attributes that are keyed with 'turn', 'bike_lane', 'crosswalk',
        'separate_path', 'speed_limit', 'signalized', or 'traffic_volume'
        
        Returns:
            dict: dictionary of randomly generated intersection data

        TODO: only generate for missing attributes
        """
        return {
            "turn": random() * 160,
            "bike_lane": choice([True, False]),
            "crosswalk": choice([True, False]),
            "separate_path": choice([True, False]),
            "speed_limit": randrange(25, 36, 5),
            "signalized": choice(["stop_sign", "traffic_light", "no_signal"]),
            "traffic_volume": random() * 1000,
        }

    def random_segment(self):
        """Creates random attributes for a segment edge in a DiGraph
        
        random_segment does not check for already existing attributes in the edge and will subsequently
        overwrite any data attributes that are keyed with 'bike_lane', 'separate_path',
        'speed_limit', or 'traffic_volume'

        Returns:
            dict: dictionary of randomly generated segment data

        TODO: only generate for missing attributes
        """
        return {
            "bike_lane": choice([True, False]),
            "separate_path": choice([True, False]),
            "speed_limit": randrange(25, 36, 5),
            "traffic_volume": random() * 1000,
        }

    def get_random_data(self, edge_data: dict):
        """Gets random data for either intersections or edges
        
        Calls the appropriate random data generator function by checking the
        type of street that edge_data refers to. If the edge_data is from an intersection
        it will call self.random_intersection(), if it is from a roadway segment it will
        call self.random_segment()

        Args:
            edge_data (dict): [description]
        
        Raises:
            ValueError: when edge_data 'type' is not intersection/segment or does not exist
        
        Returns:
            dict: [description]
        """
        if edge_data["type"] == "intersection":
            random_data = self.random_intersection()
        elif edge_data["type"] == "segment":
            random_data = self.random_segment()
        else:
            raise ValueError(f"Edge data({edge_data}) does not have a road 'type'")
        return random_data

    def generate_attributes(self, g: nx.DiGraph):
        """generate_attributes is a temporary function that allows us to fake having roadway attribute data
        
        Args:
            g (nx.DiGraph): networkx DiGraph
        
        Returns:
            dict : a dictionary containing attirbutes to be added to the DiGraph
        """
        attributes = {}
        for n1, n2, edge_data in g.edges(data=True):
            edge = (n1, n2)
            attributes[edge] = self.get_random_data(edge_data)
        return attributes

    @classmethod
    def add_random_attributes(cls, g: nx.DiGraph):
        """Creates a new DiGraph with randomly added roadway attribute data
        
        Args:
            g (nx.DiGraph): base networkx DiGraph that needs data
        
        Returns:
            nx.DiGraph: networkx DiGraph with random roadway attribute data
        """
        g_new = deepcopy(g)
        attributes = cls().generate_attributes(g_new)
        nx.set_edge_attributes(g_new, values=attributes)
        return g_new


def add_random_attributes(g: nx.DiGraph):
    """[summary]
    
    Args:
        g (nx.DiGraph): [description]
    
    Returns:
        [type]: [description]
    """
    return StreetDataGenerator().add_random_attributes(g)
