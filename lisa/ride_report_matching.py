import pickle
from graph import Graph, Name,  KDTreeWrapper
from matching import match_single, match_trace
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
from mire_check import get_expanded_graph_from_mire




def make_osmnx_graph(name, filepath):
    """Fetch OSM from name and pickle graph object to a file

    Args:
        name (string): name of city to fetch osm from
        filepath (string): filepath to save the object
    """
    G = Graph(Name(name))
    G.save(filepath)


def load_osmnx_graph(filepath):
    """Load psmnx graph from pickled file

    Args:
        filepath (string): filepath to pickled file
    Returns:
        osmnx graph object
    """
    pickle_in = open(filepath, "rb")
    return pickle.load(pickle_in)


def get_ride_report_paths(filepath):
    """Extract the x,y coordinates for each ride report gps signal, returned as
    a list of lists for each path

    Args:
        filepath (string): path to .geojson file

    Returns:
        list of lists of x,y coordinates
    """
    gdf = gpd.read_file(filepath)
    linestrings = gdf.geometry
    return [[coord for coord in line.coords] for line in linestrings]


def match_paths(paths, kd, G):
    """match a list of gps paths to their osmnx node ids

    Args:
        paths(list): list of list of long/lat coordinates
        kd(): kdtree object
        G: osmnx graph object
    Returns:
        list of list of osmnx node ids
    """
    matched_paths = []
    for raw_path in paths:
        match = match_trace(raw_path, kd, G.init_graph)
        matched_paths.append(match)
    expd_paths, failed_paths = G.init_paths_to_expd(
        list(filter(None, matched_paths)), False)
    if failed_paths:
        raise Exception('Found invalid path(s) during graph expansion')
    return expd_paths


def plot_overlay(df1, df2, x1, y1, x2, y2, title):
    """
    Overlay scatterplots of two dataframes

    Args:
        df1 (pandas dataframe)
        df2 (pandas dataframe)
        x1 (string): x column name of df 1
        y1 (string): y column name of df 1
        x2 (string): x column name of df 2
        y2 (string): y column name of df 2
        title (string): title of plot
    """
    ax = df1.plot(x=x1, y=y1, kind='scatter', color='b',
                  title=title)
    df2.plot(x=x2, y=y2, kind='scatter', color='g',
             ax=ax)
    plt.show()


def raw_to_df(raw):
    """Convert raw coordinates to pandas dataframe

    Args:
        raw: list of (long, lat) coordinates
    Returns:
        pandas dataframe of long/lat coordinates
    """
    return pd.DataFrame.from_records(raw, columns=['X', 'Y'])


def nodes_to_df(nodes):
    """Convert osmnx nodes to pandas dataframe of coordinates

    Args:
        nodes: list of osmnx node ids
    Returns:
        pandas dataframe of long/lat coordinates
    """
    coords = []
    for node in nodes:
        x, y = G.DiGraph.nodes[node]['x'], G.DiGraph.nodes[node]['y']
        coords.append((x, y))
    return pd.DataFrame.from_records(coords, columns=['x', 'y'])


if __name__ == "__main__":

    # make_osmnx_graph("Washington DC",
    #                  '/home/udesai/SCOPE/lisa/lisa/dc.pickle')
    # G = load_osmnx_graph('dc.pickle')


    gdb = "scratch_022819.gdb"
    node_layer = 3
    edge_layer = 2
    G = get_expanded_graph_from_mire(gdb, node_layer, edge_layer)
    kd = KDTreeWrapper(G.init_graph)

    # import osmnx as ox
    # ox.plot_graph(G.init_graph)

    matched_pickle_paths = []

    # paths = get_ride_report_paths('RideReportRoutes.geojson')
    import os
    import pickle

    PATH = "Z:\\SCOPE_Teams_2018-19\\Volpe_Santos\\data\\ddot\\processed\\"

    data_files = os.listdir(PATH)

    os.chdir("Z:\\")

    for data_file in data_files:
        filename = PATH + data_file

        if data_file.endswith(".csv"):
            pass
        elif data_file.endswith(".p"):
            try:
                with open(str(filename), "rb") as fp:

                    tmp = pickle.load(fp)
                    # print(tmp[0])
                    tmp = [[(tup[1],tup[0]) for tup in route[3]] for route in tmp]

                    print(tmp[0])

                    res = match_paths(tmp, kd, G)

    
                    matched_pickle_paths.append(res)

            except Exception as ex:
                print(3, ex, filename)
                continue

    print(matched_pickle_paths[0])

    with open("Z:\\SCOPE_Teams_2018-19\\Volpe_Santos\\data\\ddot\\processed\\ride_report_matching.final", "wb") as fp:
        pickle.dump(matched_pickle_paths, fp)


    # res = match_paths(paths, kd, G)





    # raw = paths[0]
    # nodes = match_trace(raw, kd, G.init_graph)
    # nodes = G.init_path_to_expd(nodes)
    # node_df = nodes_to_df(nodes)
    # raw_df = raw_to_df(raw)
    # plot_overlay(raw_df, node_df, 'X', 'Y', 'x', 'y', 'blue raw nodes, green osmnx nodes')
