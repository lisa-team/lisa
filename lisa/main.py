# -*- coding: utf-8 -*-
import calibrate
import matching
import graph
import mire_check
import sys
import os
import pickle
import pprint
import stream_routes_from_csv as stream_routes

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


def create_graph_and_kd_tree(gdb, node_layer, edge_layer, out_dir, save_or_load = "save", graph_name = "traffic_graph.pickle", kd_name = "kd_tree.pickle"):
    G = None
    kd = None

    if save_or_load == "save":

        G = mire_check.get_expanded_graph_from_mire(gdb, node_layer, edge_layer)
        kd = graph.KDTreeWrapper(G.init_graph)

        with open(graph_name, "wb+") as g_n:
            pickle.dump(G, g_n)

        with open(kd_name, "wb+") as k_n:
            pickle.dump(kd, k_n)

    elif save_or_load == "load":

        with open(graph_name, "rb") as g_n1:
            G = pickle.load(g_n1)

        with open(kd_name, "rb") as k_n1:
            kd = pickle.load(k_n1)

    return G, kd



#TODO: If the total size of gps data csvs is too large, the program will fail from MemoryError. Might require rewrite to pickle/save intermediate results.
# paths_from_csvs/res should be the first intermediate result, followed by df_intersections and df_segments.
if __name__ == "__main__":
    gdb = sys.argv[1]
    PATH = sys.argv[2] # path to csvs containing gps data

    OUT_DIR = None

    if len(sys.argv)>3:
        OUT_DIR = sys.argv[3]


    node_layer = 3
    edge_layer = 2

    graph_name = "traffic_graph.pickle"
    kd_name = "kd_tree.pickle"

    G, kd = create_graph_and_kd_tree(gdb, node_layer, edge layer, OUT_DIR, True, graph_name, kd_name)

    data_files = os.listdir(PATH)
    os.chdir(PATH)

    paths_from_csvs = []

    # Match the GPS coordinates to graph nodes 
    for data_file in data_files:

        if data_file.endswith(".csv"):

            filename = PATH + data_file

            routes_lat_long = stream_routes.get_routes_from_single_csv(filename, debug = False)

            if routes_lat_long:

                routes_long_lat = [[(tup[1],tup[0]) for tup in route] for route in routes_lat_long]

                paths_from_csv = matching.match_paths(routes_long_lat, kd, G)

                # paths_from_csv is a list of paths from a single csv. Each path is a list of integers which correspond to node IDs in our traffic graph.

                paths_from_csvs.extend(paths_from_csv)

            else:

                print("No routes from", filename)



    res = paths_from_csvs

    res_name = "routes_from_all_csvs.pickle"

    os.chdir(OUT_DIR)

    with open(res_name, "wb+") as r_n:
        pickle.dump(res, r_n)

        
    # print("res:\n", res)

    if res:
        featurelist_intersections = ['distance_efficiency','stops', 'signal'] 
        featurelist_segments = ['distance_efficiency', 'RoadType', 'Shape_Length']

        (df_intersections, df_segments) = calibrate.create_dataframes(G.DiGraph, res, (featurelist_intersections, featurelist_segments))

        fit_summary_intersections, summary_dict_intersections = calibrate.create_model(df_intersections, featurelist_intersections)

        fit_summary_segments, summary_dict_segments = calibrate.create_model(df_segments, featurelist_segments)

        pprint.pprint(summary_dict_intersections)
        pprint.pprint(summary_dict_segments)

    else:
        print("No data at all")