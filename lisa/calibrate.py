# -*- coding: utf-8 -*-
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
import math
from mire_check import *
from ride_report_matching import *
from calibrate_helper_functions import *

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
    
    intersections_featurelist = ['distance_efficiency', 'notAtGrade', 'stops', 'signal', 'pedsignal', 'rr', 'yield']
    featurelist = ['distance_efficiency', 'has_comp', 'notAtGrade', 'stops', 'signal', 'pedsignal', 'rr', 'yield']

    (df_intersections, _, _) = create_dataframes(G, res, intersections_featurelist)
    print("dataframe:\n", df_intersections[:15])
    fit_summary, summary_dict = create_model(df_intersections, intersections_featurelist)
    
    import pprint
    pprint.pprint(fit_summary)
    pprint.pprint(summary_dict)
    print(sort_input_attributes(summary_dict))

    