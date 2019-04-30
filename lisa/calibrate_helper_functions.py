# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from graph import *
from random import random, randrange, choice
import math
import pprint

def calc_distance_efficiency(G: nx.DiGraph, edgenode1: NodeID, edgenode2: NodeID, endnode: NodeID):
    """Takes in an edge and the endpoint to calculate how well the edge choice leads to the endpoint
        Does this by calculating the distance between:
            - first point to the end
            - second point to the end
            - length of the edge

        returns ((first point to the end) - (second point to the end))/(length of edge)
    Args:
        edgenode1 (NodeID): Starting node identifier for the given edge
        edgenode2 (NodeID): Ending node identifier for the given edge
        endnode (NodeID): Final node identifier for the last node in the path

    Returns:
        efficiency_calculation (float): numerical representation of the progress towards the ending node
    """
    
    (edgenode1_x, edgenode1_y) = G.nodes[edgenode1]['x'], G.nodes[edgenode1]['y']
    (edgenode2_x, edgenode2_y) = G.nodes[edgenode2]['x'], G.nodes[edgenode2]['y']
    (endnode_x,     endnode_y) = G.nodes[endnode]['x'],     G.nodes[endnode]['y']
    
    distance_edgenode1_edgenode2    = np.sqrt(np.square(edgenode1_x - edgenode2_x)  + np.square(edgenode1_y -   edgenode2_y))
    distance_edgenode1_end          = np.sqrt(np.square(edgenode1_x - endnode_x)    + np.square(edgenode1_y -   endnode_y))
    distance_edgenode2_end          = np.sqrt(np.square(endnode_x   - edgenode2_x)  + np.square(endnode_y   -   edgenode2_y))

    progress_to_end = distance_edgenode1_end - distance_edgenode2_end
    
    efficiency_calculation = (progress_to_end/distance_edgenode1_edgenode2)
    
    return efficiency_calculation

def find_attribute_dict(G: nx.DiGraph, node1: NodeID, node2: NodeID, endnode: NodeID, featurelist: list):
    """Returns the attributes of an edge from node1 to node2 in the context of G and endnode
    
    Args:
        G (nx.DiGraph): DiGraph representation of the road network
        node1 (NodeID): Starting node identifier for the given edge
        node2 (NodeID): Ending node identifier for the given edge
        endnode (NodeID): Node identifier of last node in the path for distance calculation
        featurelist (list): list of features that will be considered for model fitting.

    Returns:
        attributes (dict): dictionary with an value for each feature in the featurelist
    """
    edgeID              = (node1, node2)
    current_edge_data   = dict(G.edges[edgeID])
    attributes          = {}

    if not current_edge_data:
        print("NO EDGE DATA FOR ATTR DICT")
    else:
        print("CURRENT_EDGE_DATA", current_edge_data)
    
    try:
        for feature in featurelist:
            if (feature == "distance_efficiency"):
                feature_value = calc_distance_efficiency(G, node1, node2, endnode)

            else:
                feature_value = current_edge_data['attributes'][feature]
            attributes[feature] = feature_value

        
        assert (len(attributes) == len(featurelist)), "attributes length: " + str(len(attributes)) + " featurelist length: " + str(len(featurelist))
        attributes["type"] = current_edge_data["type"]


        return attributes

    except Exception as ex:
        print("EXCEPTION:", ex)
        return



def check_path_validity(G: nx.DiGraph, path):
    """Takes in a list of nodes and loops through each one to see if it is valid. 

    
    Args:
        path ([nx.nodeID]): List of Node IDs needed 
    """
    for node_index in range(len(path)-1):
        current_node = path[node_index]
        next_node = path[node_index + 1]

        print("CURRENT NODE TYPE:", type(current_node))
        
        current_node_neighbors = set(G.neighbors(current_node))
        if next_node not in current_node_neighbors:
            print("node_index: ", node_index)
            print(current_node, next_node, current_node_neighbors)
            return False

    return True

def check_for_data(G, current_node, neighbors, end_node, segment_features, intersection_features):
    good_attributes = True

    has_segment_attribute_dict      = None 
    has_intersection_attribute_dict = None

    # check if there are attributes in neighboring segments and neighboring intersections:
    for neighbor in neighbors:

        neighbors_of_neighbor = list(G.neighbors(neighbor))

        neighbor_of_neighbor = neighbors_of_neighbor[0]
        neighbors_of_neighbor_of_neighbor = list(G.neighbors(neighbor_of_neighbor))


        if (neighbor_of_neighbor and neighbors_of_neighbor_of_neighbor):

            has_segment_attribute_dict = find_attribute_dict(G, neighbor, neighbor_of_neighbor, end_node, segment_features)
            
            has_intersection_attribute_dict = find_attribute_dict(G, neighbor_of_neighbor, neighbors_of_neighbor_of_neighbor[0], end_node, intersection_features)

        data_present = (has_segment_attribute_dict and has_intersection_attribute_dict)
        
        if not (data_present and (has_segment_attribute_dict["type"] == "segment") and (has_intersection_attribute_dict["type"] == "intersection")):
            good_attributes = False
            break
    
    return good_attributes
