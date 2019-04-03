import pickle
from graph import Graph, Name,  KDTreeWrapper
from matching import match_single, match_trace
import geopandas as gpd


def get_ride_report_paths():
    """
    Extract the x,y coordinates for each ride report gps signal, returned as a
    list of lists for each path
    """
    gdf = gpd.read_file('RideReportRoutes.geojson')
    linestrings = gdf.geometry
    return [[coord for coord in line.coords] for line in linestrings]


def make_osmnx_graph(name, filepath):
    """
    Fetch OSM from name and pickle graph object to a file
    :param name: name of city to fetch osm from
    :param filepath: filepath to save the object
    :type filepath: string
    """
    G = Graph(Name(name))
    G.save(filepath)


def load_osmnx_graph(filepath):
    """
    Unpickle the osm graph object
    :param filepath: filepath to the object
    :type filepath: string
    :rtype: Graph_Wrapper.Graph
    """
    pickle_in = open(filepath, "rb")
    return pickle.load(pickle_in)


if __name__ == "__main__":

    # make_osmnx_graph("Washington DC",
    #                  '/home/udesai/SCOPE/lisa/lisa/dc.pickle')
    G = load_osmnx_graph('dc.pickle')
    kd = KDTreeWrapper(G.DiGraph)
    paths = get_ride_report_paths()
    single = match_single(paths[0][0], kd)
