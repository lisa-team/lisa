# -*- coding: utf-8 -*-
""" A collection of functions used for data import/export and configuration.

This module contains the functions required for manipulating input data for use
in the rest of the package, and also export to standard formats.

Example:
    TODO insert some examples if needed

"""

# TODO some of what Nathan, Uma, and Annie have been writing goes here
from random import random, randrange, choice
import networkx as nx
from copy import deepcopy

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
            attributes.update(g.edges(data=True)[edge])
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