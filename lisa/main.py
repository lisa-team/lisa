# -*- coding: utf-8 -*-
import calibrate
import matching
import graph
import mire_check
import sys
import os
import pickle
import pprint
import stream_routes_from_csv

""" This script runs through all of the processes: infrastructure graph expansion, route-matching, and input calibration
    To run this script you need 2 command line arguments: the path to the MIRE data (a .gdb file) and the path to processed bikeshare routes directory

    Here's what an input would look like:
    $ python main.py $path_to_infrastructure_data$ $path_to_bikeshare_routes$

    The output would look something like this:

    {'distance_efficiency': {'p_values': float,
                         'parameters': float,
                         'robust_p_values': float,
                         'robust_std_err': float,
                         'robust_t_stats': float,
                         'std_err': float,
                         't_stats': float},
    'signal': {'p_values': float,
                'parameters': float,
                'robust_p_values': float,
                'robust_std_err': float,
                'robust_t_stats': float,
                'std_err': float,
                't_stats': float},
    'stops': {'p_values': float,
            'parameters': float,
            'robust_p_values': float,
            'robust_std_err': float,
            'robust_t_stats': float,
            'std_err': float,
            't_stats': float}}
    {'RoadType': {'p_values': float,
                'parameters': float,
                'robust_p_values': float,
                'robust_std_err': float,
                'robust_t_stats': float,
                'std_err': float,
                't_stats': float},
    'Shape_Length': {'p_values': float,
                    'parameters': float,
                    'robust_p_values': float,
                    'robust_std_err': float,
                    'robust_t_stats': float,
                    'std_err': float,
                    't_stats': float},
    'distance_efficiency': {'p_values': float,
                            'parameters': float,
                            'robust_p_values': float,
                            'robust_std_err': float,
                            'robust_t_stats': float,
                            'std_err': float,
                            't_stats': float}}

    If you want to add attributes to your analysis, you can add categories to featurelist_intersections and featurelist_segments if the category's attribute is already in the infrastructure graph
"""

if __name__ == "__main__":

    gdb = sys.argv[1]
    node_layer = 3
    edge_layer = 2




    G = mire_check.get_expanded_graph_from_mire(gdb, node_layer, edge_layer)

    kd = graph.KDTreeWrapper(G.init_graph)


    matched_paths = []

    all_paths = []

    PATH = sys.argv[2]

    data_files = os.listdir(PATH)

    os.chdir(PATH)

    node_sum = 0
    num_nodes = 0



    # Match the GPS coordinates to graph nodes 
    for data_file in data_files:
        filename = PATH + data_file

        if data_file.endswith(".csv"):
            raw_paths = stream_routes_from_csv.get_routes_from_csv(filename)

            flipped_paths = [[(tup[1], tup[0]) for tup in path] for path in raw_paths] # flips lat,long to long,lat because match_paths takes in routes of long,lat

            matched_paths = matching.match_paths()

    
    for file_routes in matched_paths:
        all_paths.extend(file_routes)

    featurelist_intersections = ['distance_efficiency','stops', 'signal'] 
    featurelist_segments = ['distance_efficiency', 'RoadType', 'Shape_Length']

    (df_intersections, df_segments) = calibrate.create_dataframes(G.DiGraph, all_paths, (featurelist_intersections, featurelist_segments))

    fit_summary_intersections, summary_dict_intersections = calibrate.create_model(df_intersections, featurelist_intersections)

    fit_summary_segments, summary_dict_segments = calibrate.create_model(df_segments, featurelist_segments)

    pprint.pprint(summary_dict_intersections)
    pprint.pprint(summary_dict_segments)