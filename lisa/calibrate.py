# -*- coding: utf-8 -*-
import pylogit as pl
from graph import *
from data import StreetDataGenerator
from collections import OrderedDict
from nx_types import *
from mire_check import *
from ride_report_matching import *
from calibrate_helper_functions import *
    
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
    n_intersection_feats            = len(intersection_features)
    choice_features_intersections   = []
    observation_ids                 = []
    choice_indicators               = []
    choice_ids                      = []
    observation_id                  = 0

    for path_index in range(len(paths)):
        
        path = paths[path_index]

        if check_path_validity(G, path):

            for i in range(len(path)-1):
                print("\n\n\n\n")
                current_node = path[i]
                end_node = path[-1]
                neighbors = list(G.neighbors(current_node))
                n_choices = len(neighbors)

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
                        
                        has_intersection_attribute_dict = find_attribute_dict(G, neighbor_of_neighbor, neighbors_of_neighbor_of_neighbor[-1], end_node, intersection_features)

                    data_present = (has_segment_attribute_dict and has_intersection_attribute_dict)
                    
                    if (not data_present):
                        print("data not present")
                        good_attributes = False
                    else:
                        good_attributes = (has_segment_attribute_dict["type"] == "segment") and (has_intersection_attribute_dict["type"] == "intersection")


                # attempting to add to the intersections dataframe ONLY if # choices > 1 and there's data in the attributes section
                if ((n_choices > 1) and (good_attributes)):
                    
                    observation_ids.append(observation_id*np.ones((n_choices,))) 
                    
                    # 'i' is the "index" of the observation, or the reason why we know which observation is which
                    for neighbor_i in range(len(neighbors)):
                        
                        neighbor = neighbors[neighbor_i]
                        
                        
                        neighbors_of_neighbor = list(G.neighbors(neighbor))
                        neighbor_of_neighbor = neighbors_of_neighbor[0]
                        
                        
                        neighbors_of_neighbor_of_neighbor = list(G.neighbors(neighbor_of_neighbor))
                        
                        print("current_node: ", current_node)
                        print("current node's neighbor ", neighbor)
                        print("current node's neighbor's neighbors: ", neighbors_of_neighbor_of_neighbor)
                        print("EdgeID: ", (neighbor_of_neighbor, neighbors_of_neighbor_of_neighbor[0]))
                        current_intersection_data = find_attribute_dict(G, neighbor_of_neighbor, neighbors_of_neighbor_of_neighbor[0], end_node, intersection_features)


                        # collect all the observations for all neighbors
                        current_observation_choice_features = []

                        # iteratively adding each feature value to the observation
                        for feature in intersection_features:
                            feature_value = current_intersection_data[feature] 
                            current_observation_choice_features.append(feature_value)

                        # append each possibility's features to the dataframe
                        choice_features_intersections.append(current_observation_choice_features)

                    # marking the choiceID's choice as '1' among zeros
                    choice_indicators.append(np.zeros((n_choices,)))
                    chosen = neighbors.index(path[i+1])
                    choice_indicators[-1][chosen] = 1
                    
                    # All the possible choices out at this observation:
                    choice_ids.append(np.arange(n_choices))
                    observation_id += 1
        else:
            print('Path # ' + str(path_index) + ' is not possible in the MIRE graph')
    
    # preparing columns for the dataframe (long) format
    overall_observation_ids                 = np.concatenate(observation_ids) # should have the same # of observations for both datasets...
    choice_features_overall_intersections   = np.vstack(choice_features_intersections) # different choice features for all
    overall_choice_indicators               = np.concatenate(choice_indicators) # same choice indicators for both...
    overall_choice_ids                      = np.concatenate(choice_ids) # should be same because it's alt_id, or assigning IDs to 

    df_intersections    = pd.DataFrame()
    df_segments         = pd.DataFrame()
    
    df_intersections['obs_ids'] = overall_observation_ids
    df_intersections['choices'] = overall_choice_indicators
    df_intersections['alt_ids'] = overall_choice_ids
    
    for i in range(n_intersection_feats):
        spec = intersection_features[i]
        df_intersections[spec] = choice_features_overall_intersections[:,i]

    return (df_intersections, df_segments)



def create_model(dataframe: pd.DataFrame, feature_list: list):
    """Fitting multinomial logit model to the dataframe choices with specifications
    
    Args:
        dataframe (pd.DataFrame): Dataframe representation of the routes and the choices taken in our data
        feature_list (list[str]): list of (id#, featurename) tuples present in G.edges output
        
        
        specs (OrderedDict): Each category with an 'all_same' value to make sure the column ID does not change for the same attribute
        spec_names (OrderedDict): Customizable specs for each category for nesting cases
    
    Returns:
        (fit_summary, summary): fit_summary is a dictionary that shows the overall model's fit, 
                                summary is a dictionary that shows each input's fit in the model
    """
    
    # This just means the columns will be consistent across choices
    spec_names = OrderedDict()
    specs = OrderedDict()
    for i in range(len(feature_list)):
        spec = feature_list[i]
        spec_names[spec] = spec
        specs[spec] = 'all_same'

    assert (len(specs) == len(spec_names))

    # Fit to a multinomial logit model (MNL)
    choice_model = pl.create_choice_model(data=dataframe,
                                            alt_id_col='alt_ids',
                                            obs_id_col='obs_ids',
                                            choice_col='choices',
                                            specification=specs,
                                            model_type="MNL",
                                            names=spec_names)
    
    choice_model.fit_mle(np.zeros(len(feature_list)), print_res = False)
    
    fit_summary_print_output = choice_model.fit_summary
    summary_print_output = choice_model.summary

    summary = (summary_print_output.to_dict(orient="index"))

    fit_summary = (fit_summary_print_output.to_dict())

    choice_model.get_statsmodels_summary()

    return fit_summary, summary


# TODO: Combine sort_input_attributes with weights_to_lts
def sort_input_attributes(summary_dict: dict):
    """Takes in the fitted model summary and picks the statistically significant input variables

    Args:
        summary_dict (dict): 2nd output from from create_model() method
    
    Returns:
        weights (dict): maps variables to their weights
    """
    weights = {}

    # try to reject any variables not worth considering
    for attribute in summary_dict:
        std_error = summary_dict[attribute]['std_err']
        weight = summary_dict[attribute]['parameters']
        p_value = summary_dict[attribute]['p_values']
        
        if (p_value < 0.10):
            weights[attribute] = (weight, std_error)
    
    return weights

def weights_to_lts(attribute_weights: dict):
    """ 
    
    Args:
        attribute_weights (dict): [description]
    """

    pass
    


if __name__ == "__main__":

    gdb = "scratch_022819.gdb"
    node_layer = 3
    edge_layer = 2


    G = get_expanded_graph_from_mire(gdb, node_layer, edge_layer).DiGraph

    nodes = list(G.nodes)

    # generate paths (replace with map-matching paths later)
    res = []
    for i in range(1):
        start = choice(nodes)
        end = choice(nodes)
        try:
            path = nx.shortest_path(G, start, end)
            res.append(path)

        except nx.exception.NetworkXNoPath as e:
            print(e)
    # print("res: ", res)

    # approach: keep all the variables in here for now, then remove the ones that are linearly dependent
    featurelist_intersections = ['distance_efficiency', 'grade', 'notAtGrade','stops', 'signal',  'pedsignal', 'rr', 'yield'] 
    featurelist_segments = ['distance_efficiency', 'RoadType', 'Directionality', 'Shape_Length']

    (df_intersections, _) = create_dataframes(G, res, (featurelist_intersections, featurelist_segments))
    # print("dataframe:\n", df_intersections[:20])


    fit_summary_intersections, summary_dict_intersections = create_model(df_intersections, featurelist_intersections)
    
    # import pprint
    # pprint.pprint(fit_summary_intersections)
    # pprint.pprint(summary_dict_intersections)
    # # print(sort_input_attributes(summary_dict))

    