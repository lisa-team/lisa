import pylogit as pl
import pandas as pd
import numpy as np
from graph import *
import random
from collections import OrderedDict
import sys
from random import random, randrange, choice


def find_attribute_dict(G, startNode, endNode):
    """
        Gets attributes of the edge between two nodes
    """
    edgeID = (startNode, endNode)
    return dict(G.edges[edgeID])
    
def create_dataframe(G, paths, feature_list):
    """
    Loops through each node in the path and gets the considered features in the dataframe
    """
    n_feats = len(feature_list)
    choice_features = []
    observation_ids = []
    choice_indicators = []
    choice_ids = []

    for path_index in range(len(paths)):
        path = paths[path_index]

        for i in range(len(path)-1):

            # append all the edge options to a list of lists called choice_features
            current_node = path[i]
            neighbors = list(G.neighbors(current_node))
            n_choices = len(neighbors)
            observation_ids.append(i*np.ones((n_choices,))) 
            # 'i' is the "index" of the observation, or the reason why we know which observation is which
            for neighbor in neighbors:
                current_attribute_dict = find_attribute_dict(G, current_node, neighbor)
                # iteratively adding each feature value to the observation
                current_observation_choice_features = []
                for feature in feature_list:
                    current_observation_choice_features.append(current_attribute_dict[feature])
                choice_features.append(current_observation_choice_features)

            # marking the choiceID's choice as '1' among zeros
            choice_indicators.append(np.zeros((n_choices,)))
            chosen = neighbors.index(path[i+1])
            choice_indicators[-1][chosen] = 1
            # All the possible choices out at this observation:
            choice_ids.append(np.arange(n_choices))

    # preparing columns for the dataframe (long) format
    overall_observation_ids = np.concatenate(observation_ids)
    choice_features_overall = np.vstack(choice_features)
    overall_choice_indicators = np.concatenate(choice_indicators)
    overall_choice_ids = np.concatenate(choice_ids)

    df = pd.DataFrame()

    df['obs_ids'] = overall_observation_ids
    df['choices'] = overall_choice_indicators
    df['alt_ids'] = overall_choice_ids

    # print(df[['obs_ids','choices', 'alt_ids']])
    #     obs_ids  choices  alt_ids
    # 0       0.0      0.0        0
    # 1       0.0      1.0        1
    # 2       1.0      1.0        0
    # 3       2.0      0.0        0
    # 4       2.0      1.0        1
    # 5       3.0      1.0        0
    # 6       4.0      1.0        0
    # 7       4.0      0.0        1
    # 8       5.0      1.0        0
    # 9       6.0      1.0        0
    # 10      6.0      0.0        1
    # 11      7.0      1.0        0
    # 12      8.0      0.0        0
    # 13      8.0      1.0        1
    # 14      8.0      0.0        2

    # Ignore the leftmost column. As you see, each observation is a "choice". 
    # The # of rows w/ the same observation = # of choices the biker makes
    # The choices are either 1 or 0. Each observation makes a 

    # The next few lines just mean the columns will be consistent across choices
    spec_names = OrderedDict()
    specs = OrderedDict()
    for i in range(n_feats):
        spec = feature_list[i]
        spec_names[spec] = spec
        specs[spec] = 'all_same'
        df[spec] = choice_features_overall[:,i]
    return (df, n_feats, specs, spec_names)

def create_model(dataframe,n_feats, specs, spec_names):
    # Estimate the multinomial logit model (MNL)
    choice_model = pl.create_choice_model(data=df,
                                            alt_id_col='alt_ids',
                                            obs_id_col='obs_ids',
                                            choice_col='choices',
                                            specification=specs,
                                            model_type="MNL",
                                            names=spec_names)

    # Specify the initial values and method for the optimization.
    choice_model.fit_mle(np.zeros(n_feats))

    # Look at the estimation results
    # choice_model.get_statsmodels_summary()

    # Look at other all results at the same time
    choice_model.print_summaries()



    # Look at the general and goodness of fit statistics
    choice_model.fit_summary
    return





if __name__ == "__main__":

    # bbox = Bbox(38.88300016, 38.878726840000006, -77.09939832, -77.10500768)
    # G = Graph(bbox)
    # G.save("boundgraph")

    G = Graph.from_file("boundgraph").DiGraph
    nodes = list(G.node)
    # print(type(nodes))
    start = choice(nodes)
    end = choice(nodes)
    # print("start, end: ", start, end)
    path = nx.shortest_path(G, start, end)
    print("path: ", path)
    (df, n_feats, specs, spec_names) = create_dataframe(G, [path], ["traffic_volume", "speed_limit"])
    create_model(df, n_feats, specs, spec_names)