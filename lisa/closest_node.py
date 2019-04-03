import math
import pickle
from graph import KDTreeWrapper


def nearest_node(coord, kd, G, n=3):
    """
    Find the nearest osmnx node to a long, lat coordinate.
    Does this by querying the closest n nodes, finding all adjacent edges,
    calculating the closest edge, and returning the relevant node.

    Args:
        coord (Tuple[float, float]): the latitude and longitude to match
        kd (graph.KDTreeWrapper object): the kd tree built off init_graph
        G (Graph object): osmnx digraph
        n (int): number of closest nodes to consider
    Returns:
        a (float): a in ax + by + c = 0
        b (float): b in ax + by + c = 0
        c (float): c in ax + by + c = 0
    """
    nodes, distances = kd.query_min_dist_nodes(coord, n)
    min_dist = math.inf
    min_node = nodes[0]
    for node in nodes:
        for neighbor in G.neighbors(node):
            a, b, c = generate_line(G.nodes[node],
                                    G.nodes[neighbor])
            d = calculate_dist(coord, a, b, c)
            if d < min_dist:
                min_dist = d
                min_node = node
    return min_node, min_dist


def generate_line(node1, node2):
    """
    Find the standard form equation of a line given two osmnx nodes.

    Args:
        node1 (int): osmnx node id
        node2 (int): osmnx node id
    Returns:
        a (float): a in ax + by + c = 0
        b (float): b in ax + by + c = 0
        c (float): c in ax + by + c = 0
    """
    a = node1['y'] - node2['y']
    b = node2['x'] - node1['x']
    c = a*(node2['x']) + b*(node2['y'])
    return a, b, -c


def calculate_dist(coord, a, b, c):
    """
    Calculate the distance d from a point (x0, y0) to the line ax + by + c = 0,
    defined as the shortest distance between a fixed point and any point on the
    line. It is the length of the line segment that is perpendicular to the
    line and passes through the point.

    Args:
        coord (Tuple[float, float]): the latitude and longitude to match
        a (float): a in ax + by + c = 0
        b (float): b in ax + by + c = 0
        c (float): c in ax + by + c = 0
    Returns:
        d (float): distance
    """
    x0 = coord[0]
    y0 = coord[1]
    d = abs(a*x0 + b*y0 + c)/math.sqrt(a*a + b*b)
    return d


if __name__ == "__main__":

    pickle_in = open('dc.pickle', "rb")
    G = pickle.load(pickle_in)
    kd = KDTreeWrapper(G.DiGraph)
    node = nearest_node((38.91, 77.04), kd, G.DiGraph)
