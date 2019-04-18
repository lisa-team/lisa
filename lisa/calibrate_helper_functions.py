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
    (endnode_x, endnode_y) = G.nodes[endnode]['x'], G.nodes[endnode]['y']
    
    distance_edgenode1_edgenode2 = np.sqrt(np.square(edgenode1_x - edgenode2_x) + np.square(edgenode1_y - edgenode2_y))
    distance_edgenode1_end = np.sqrt(np.square(edgenode1_x - endnode_x) + np.square(edgenode1_y - endnode_y))
    distance_edgenode2_end = np.sqrt(np.square(endnode_x - edgenode2_x) + np.square(endnode_y - edgenode2_y))

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
    edgeID = (node1, node2)
    current_edge_data = dict(G.edges[edgeID])
    attributes = {}
    
    try:
        for feature in featurelist:
            if (feature == "distance_efficiency"):
                feature_value = calc_distance_efficiency(G, node1, node2, endnode)

            else:
                feature_value = current_edge_data['attributes'][feature]
            attributes[feature] = feature_value
        
        assert (len(attributes) == len(featurelist)), "attributes length: " + str(len(attributes)) + " featurelist length: " + str(len(featurelist))

        return attributes
    
    except Exception:
        return

    
def create_dataframes(G: nx.DiGraph, paths: list, featurelists: tuple):
    """Create the pandas dataframe that is compatible with pylogit
    
    Args:
        G (nx.DiGraph): DiGraph representation of the road network
        paths (list[NodeID]): chosen list of nodeIDs the biker visited
        featurelists tuple(list): list of features that will be considered for model fitting. Structured like (intersections list, segments list)
    
    Returns:
        (df_intersections, df_segments, df_turns): Pandas dataframes needed to run intersections, segments, and turns pylogit model
    """

    (intersection_features, segment_features) = featurelists


    # INTERSECTION features!!
    n_feats = len(intersection_features)
    choice_features = []
    observation_ids = []
    choice_indicators = []
    choice_ids = []
    observation_id = 0

    for path_index in range(len(paths)):
        
        path = paths[path_index]

        for i in range(len(path)-1):

            # append all the edge options to a list of lists called choice_features
            current_node = path[i]
            end_node = path[-1]
            neighbors = list(G.neighbors(current_node))
            n_choices = len(neighbors)

            has_attributes = True

            # check if there are attributes from this node to its neighbors:
            for neighbor in neighbors:
                edgeID = (current_node, neighbor)
                has_attribute_dict = find_attribute_dict(G, current_node, neighbor, end_node, intersection_features)
                
                if (not has_attribute_dict):
                    has_attributes = False
            

            # attempting to add to the intersections dataframe ONLY if # choices > 1 and there's data in the attributes section
            if ((n_choices > 1) and (has_attributes)):
            # if (has_attributes):    
                observation_ids.append(observation_id*np.ones((n_choices,))) 
                
                # 'i' is the "index" of the observation, or the reason why we know which observation is which
                for neighbor in neighbors:
                    
                    edgeID = (current_node, neighbor)
                    intersection_attribute_dict = find_attribute_dict(G, current_node, neighbor, end_node, intersection_features)
                    # type = dict(G.edges[edgeID])['type'] # line that might be useful for later

                    # collect all the observations for all neighbors
                    current_observation_choice_features = []

                    # iteratively adding each feature value to the observation
                    for feature in intersection_features:
                        current_feature = intersection_attribute_dict[feature]
                        current_observation_choice_features.append(current_feature)
                    
                    # append each possibility's features to the dataframe
                    choice_features.append(current_observation_choice_features)

                # marking the choiceID's choice as '1' among zeros
                choice_indicators.append(np.zeros((n_choices,)))
                chosen = neighbors.index(path[i+1])
                choice_indicators[-1][chosen] = 1
                
                # All the possible choices out at this observation:
                choice_ids.append(np.arange(n_choices))
                observation_id += 1
    
    # preparing columns for the dataframe (long) format
    overall_observation_ids = np.concatenate(observation_ids)
    choice_features_overall = np.vstack(choice_features)
    overall_choice_indicators = np.concatenate(choice_indicators)
    overall_choice_ids = np.concatenate(choice_ids)

    df_intersections = pd.DataFrame()
    df_segments = pd.DataFrame()
    
    df_intersections['obs_ids'] = overall_observation_ids
    df_intersections['choices'] = overall_choice_indicators
    df_intersections['alt_ids'] = overall_choice_ids
    
    for i in range(n_feats):
        spec = intersection_features[i]
        df_intersections[spec] = choice_features_overall[:,i]

    return (df_intersections, df_segments)