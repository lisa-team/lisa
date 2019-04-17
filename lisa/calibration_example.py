import pylogit as pl
import pandas as pd
import numpy as np
from graph import *
from data import StreetDataGenerator
import random
from collections import OrderedDict
import sys
from random import random, randrange, choice
from nx_types import *

def calc_distance_efficiency(edgenode1: NodeID, edgenode2: NodeID, endnode: NodeID):
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

def find_attribute_dict(G: nx.DiGraph, node1: NodeID, node2: NodeID, endnode: NodeID):
    """Returns the attributes of an edge from node1 to node2 in the context of G and endnode
    
    Args:
        G (nx.DiGraph): DiGraph representation of the road network
        node1 (NodeID): Starting node identifier for the given edge
        node2 (NodeID): Ending node identifier for the given edge
        endnode (NodeID): Final node identifier for the last node in the path to help with distance calculation
    
    Returns:
        attribute_dict (dict): 
    """
    edgeID = (node1, node2)
    attribute_dict = dict(G.edges[edgeID])
    attribute_dict["distance_efficiency"] = calc_distance_efficiency(node1, node2, endnode)
    
    return attribute_dict
    
def create_dataframe(G, paths, feature_list):
    """
    Loops through each node in the path and gets the considered features in the dataframe
    """
    n_feats = len(feature_list)
    choice_features = []
    observation_ids = []
    choice_indicators = []
    choice_ids = []
    counter = 0

    for path_index in range(len(paths)):
        
        path = paths[path_index]

        for i in range(len(path)-1):

            # append all the edge options to a list of lists called choice_features
            current_node = path[i]
            end_node = path[-1]
            neighbors = list(G.neighbors(current_node))
            n_choices = len(neighbors)
            observation_ids.append(counter*np.ones((n_choices,))) 
            # 'i' is the "index" of the observation, or the reason why we know which observation is which
            for neighbor in neighbors:
                current_attribute_dict = find_attribute_dict(G, current_node, neighbor, end_node)
                # iteratively adding each feature value to the observation
                current_observation_choice_features = []
                
                for feature in feature_list:

                    try:
                        current_feature = current_attribute_dict[feature]
                    except KeyError as e:
                        current_feature = 0
                    current_observation_choice_features.append(current_feature)
                
                choice_features.append(current_observation_choice_features)

            # marking the choiceID's choice as '1' among zeros
            choice_indicators.append(np.zeros((n_choices,)))
            chosen = neighbors.index(path[i+1])
            choice_indicators[-1][chosen] = 1
            # All the possible choices out at this observation:
            choice_ids.append(np.arange(n_choices))
            
            counter += 1

    # preparing columns for the dataframe (long) format
    overall_observation_ids = np.concatenate(observation_ids)
    choice_features_overall = np.vstack(choice_features)
    overall_choice_indicators = np.concatenate(choice_indicators)
    overall_choice_ids = np.concatenate(choice_ids)

    df = pd.DataFrame()

    df['obs_ids'] = overall_observation_ids
    df['choices'] = overall_choice_indicators
    df['alt_ids'] = overall_choice_ids

    # The next few lines just mean the columns will be consistent across choices
    spec_names = OrderedDict()
    specs = OrderedDict()
    for i in range(n_feats):
        spec = feature_list[i]
        spec_names[spec] = spec
        specs[spec] = 'all_same'
        df[spec] = choice_features_overall[:,i]
    return (df, n_feats, specs, spec_names)

def create_model(dataframe, n_feats, specs, spec_names):
    # Fit to a multinomial logit model (MNL)
    choice_model = pl.create_choice_model(data=df,
                                            alt_id_col='alt_ids',
                                            obs_id_col='obs_ids',
                                            choice_col='choices',
                                            specification=specs,
                                            model_type="MNL",
                                            names=spec_names)
    
    # Specify the initial values and method for the optimization.
    choice_model.fit_mle(np.zeros(n_feats), print_res = False)
    fit_summary_print_output = choice_model.fit_summary
    summary_print_output = choice_model.summary

    summary = (summary_print_output.to_dict(orient="index"))

    fit_summary = (fit_summary_print_output.to_dict())

    choice_model.get_statsmodels_summary()

    print(fit_summary_print_output)
    print(summary_print_output)

    return fit_summary, summary

if __name__ == "__main__":

    # bbox = Bbox(38.88300016, 38.878726840000006, -77.09939832, -77.10500768)
    # G = Graph(bbox)
    # G.save("boundgraph")

    G = Graph.from_file("boundgraph").DiGraph
    nodes = list(G.nodes)
    paths = []
    for i in range(3):
        start = choice(nodes)
        end = choice(nodes)
        path = nx.shortest_path(G, start, end)
        paths.append(path)
    print("paths: ", paths)
    featurelist = ["bike_lane", "separate_path", "speed_limit", "traffic_volume", "crosswalk", "turn", "distance_efficiency"]
    (df, n_feats, specs, spec_names) = create_dataframe(G, paths, featurelist)
    a, b = create_model(df, n_feats, specs, spec_names)
    # print(a)
    # print(b)