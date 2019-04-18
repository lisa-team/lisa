# -*- coding: utf-8 -*-
import leather
import fiona
from fiona import transform
import os

import matplotlib.pyplot as plt

from graph import *

import networkx as nx

import pprint


def get_attributes_from_grade(grade):
    attrs = None
    if grade:
        gc = int(grade)
        attrs = {
        'grade': gc,
        'notAtGrade': 1 if gc == 1 else 0,
        'stops': 4 if gc == 12 else 2 if gc == 11 else 0,
        'signal': 1 if gc in [13, 14] else 0,
        'pedsignal': 1 if gc == 14 else 0,
        'rr': 1 if gc == 15 else 0,
        'yield': 1 if gc == 17 else 0,
        }
    return attrs


def mcoords_to_xy(mire_mcoords):
    return fiona.transform.transform(
        'EPSG:26985', 'EPSG:4326',
    [mire_mcoords[0]],
    [mire_mcoords[1]]
    ) 


def get_nodes_from_mire(gdb: str, node_layer: int):
    err = 0
    mire_coords = []
    mire_nodes = []
    with fiona.open(gdb, 'r', layer=node_layer) as inp:
        for f in inp:
            if f['properties']['INTERSECTIONID'][0] == '0':
                err+=1
                continue

            mire_mcoords = f['geometry']['coordinates']
            if mire_mcoords:

                mire_xy = mcoords_to_xy(mire_mcoords)

                if mire_xy[0][0] and mire_xy[1][0] and f['properties']['INTERSECTIONID'] != 'DCBoundary':

                    mire_nodes.append((f['properties']['INTERSECTIONID'], {"x":mire_xy[0][0],"y":mire_xy[1][0], 'attributes':get_attributes_from_grade(f['properties']['GRADE'])}))
                else:
                    err+=1
            else:
                err+=1
    return mire_nodes

def get_edges_from_mire(gdb: str, edge_layer: int):
    mire_edges = []
    err = 0
    with fiona.open(gdb, 'r', layer=edge_layer) as inp:
        for f in inp:
            firstIntersection, secondIntersection = f['properties']['FromIntersectionID'], f['properties']['ToIntersectionID']
            
            if firstIntersection[0] == '0' or secondIntersection[0] == '0' or firstIntersection == "None" or secondIntersection == "None" or firstIntersection == 'DCBoundary' or secondIntersection == 'DCBoundary':
                err +=1
                continue
            
            try:
                edge =  (firstIntersection,  secondIntersection,    {"attributes": dict(f['properties'])}   )
                edge2 = (secondIntersection, firstIntersection,     {"attributes": dict(f['properties'])}   )
                mire_edges.append(edge)
                mire_edges.append(edge2)

            except:
                err+=1
    return mire_edges

def get_initial_graph_from_mire(gdb: str, node_layer: int, edge_layer: int):
    init_graph = nx.DiGraph()

    mire_nodes = get_nodes_from_mire(gdb, node_layer)
    init_graph.add_nodes_from(mire_nodes)

    mire_edges = get_edges_from_mire(gdb, edge_layer)
    # pprint.pprint(mire_edges)
    init_graph.add_edges_from(mire_edges)


    init_graph = create_mdg(init_graph)
    
    return init_graph


def get_expanded_graph_from_mire(gdb: str, node_layer: int, edge_layer: int):
    init_graph = get_initial_graph_from_mire(gdb, node_layer, edge_layer)
    expanded_graph = Graph(bound = None, mire_graph = init_graph)
    return expanded_graph



def create_mdg(digraph):
    G = nx.MultiDiGraph()

    # hard coding graph parameters for visualization purposes
    G.graph = {
        "name": "Visualization Graph",
        "crs": {"init": "epsg:4326"},
        "simplified": True,
    }
    G.add_nodes_from(digraph.nodes(data=True))
    G.add_edges_from(
        [(n1, n2, 0, data) for n1, n2, data in digraph.edges(data=True)]
    )
    return G


if __name__ == "__main__":
    gdb = "scratch_022819.gdb"
    node_layer = 3
    edge_layer = 2

    expanded_graph = get_expanded_graph_from_mire(gdb, node_layer, edge_layer)
    
    print("expanded_graph nodes: ", list(expanded_graph.DiGraph.nodes(data=True))[0])
    print("expanded_graph edges: ", list(expanded_graph.DiGraph.edges(data=True))[:10])

    # expanded_graph.plot_graph()

