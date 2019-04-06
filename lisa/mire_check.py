import leather
import fiona
from fiona import transform
import os

import matplotlib.pyplot as plt

from graph import *

import networkx as nx


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
                edge = (firstIntersection,secondIntersection, dict(f['properties']))
                edge2 = (secondIntersection, firstIntersection, dict(f['properties']))
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

    expanded_graph.plot_graph()



    # filename = "WashingtonDCGraph.pkl"

    # get osmnx node coords
    # if not os.path.exists(filename):
    #     name = Name("Washington DC")
    #     g = Graph(name)
    #     g.save(filename)
    # else:
    #     g = Graph.from_file(filename)
    # osmnx_coords = [(a['x'], a['y']) for n, a in g.init_graph.nodes(data=True)]
    # print([k for k in g.init_graph.nodes(data=True)])



    #TODO: Make sure init_graph nodes are (nodeId), {'x':____, 'y':_____, [OPTIONAL DATA]:________ } format or whatever node_geometry needs
    # same for edges - format has to be right


    # get mire node coords
    # mire_coords = []
    # mire_nodes = []
    # mire_intersectionID_to_nodes = {}
    # temp = True
    # with fiona.open("scratch_022819.gdb", 'r', layer=3) as inp:
    #     for f in inp:
    #         if temp:
    #             print(f)
    #             # print(f['properties']['INTERSECTIONID'])
    #             # print(f['properties']['GRADE'])
    #             # print(inp)
    #             temp = False

    #         if f['properties']['INTERSECTIONID'] == '00001':
    #             continue
    #         mire_coords.append(f['geometry']['coordinates'])

    #         mire_mcoords = f['geometry']['coordinates']
    #         if mire_mcoords:

    #             mire_xy = fiona.transform.transform(
    #                 'EPSG:26985', 'EPSG:4326',
    #             [mire_mcoords[0]],
    #             [mire_mcoords[1]]
    #             )

    #             if mire_xy[0][0] and mire_xy[1][0] and f['properties']['INTERSECTIONID'] != 'DCBoundary':

    #                 mire_nodes.append((f['properties']['INTERSECTIONID'], {"x":mire_xy[0][0],"y":mire_xy[1][0], 'attributes':get_attributes_from_grade(f['properties']['GRADE'])}))
    #             else:
    #                 print(mire_xy[0][0], mire_xy[1][0])

    # # print(mire_intersectionID_to_nodes)

    # print("nodes compiled")

    # g1 = nx.DiGraph()
    # g1.add_nodes_from(mire_nodes)


    # print([k for k in g1.nodes(data=True) if k[1].get("x") is None][:10])

    # print("node adding done")

    # mire_edges = []
    # temp = True
    # err = 0
    # with fiona.open("scratch_022819.gdb", 'r', layer=2) as inp:
    #     for f in inp:
    #         if temp:
    #             print(f)
    #             temp = False
    #         firstIntersection, secondIntersection = f['properties']['FromIntersectionID'], f['properties']['ToIntersectionID']
    #         if firstIntersection[0] == '0' or secondIntersection[0] == '0' or firstIntersection == "None" or secondIntersection == "None" or firstIntersection == 'DCBoundary' or secondIntersection == 'DCBoundary':

    #             err +=1
    #             continue
    #         try:
    #             edge = (firstIntersection,secondIntersection, dict(f['properties']))
    #             mire_edges.append(edge)
    #             # print(edge)
    #         except:
    #             print("problem number", err)
    #             err+=1
    # # print("errors:", err)

    # # print("edges compiled")

    # g1.add_edges_from(mire_edges)


    # print("edge adding done")

    # print([k for k in g1.edges(data=True)][:10])
    # print("g1 nodes: ", list(g1.nodes(data=True))[0])
    # print("g1 edges: ", list(g1.edges(data=True))[0])


    # print([k for k in g1.nodes(data=True) if k[1].get("x") is None][:10])

    # g2 = Graph(bound = None, mire_graph = g1)

    # print("g2 init nodes: ", list(g2.init_graph.nodes(data=True))[0])
    # print("g2 init edges: ", list(g2.init_graph.edges(data=True))[0])

    # print("g2 nodes: ", list(g2.DiGraph.nodes(data=True))[0])
    # print("g2 edges: ", list(g2.DiGraph.edges(data=True))[:10])
    # # print("g2 node map: ", g2.node_map)



    # g2.plot_graph()



    

    # print(len(sorted(list(g2.nodes(data=True)), key=lambda n: n[0])))

    # fig, ax2 = g2.highlight_graph(edge_filter_function=None, node_filter_function=None, legend_elements=None, title = "")
    
    # plt.savefig('../out/test(1).png')
    # plt.show()


    # hover = Graph_Hover(graph=g2, fig=fig, ax=ax2)

    # hover.display_graph()

    # g = Graph(bound = None, mire_graph = g1)


    # # calculate averages
    # osmnx_centroid = (
    #     sum(map(lambda c : c[0], osmnx_coords)) / len(osmnx_coords),
    #     sum(map(lambda c : c[1], osmnx_coords)) / len(osmnx_coords)
    # )
    # mire_centroid = (
    #     sum(map(lambda c : c[0], mire_coords)) / len(mire_coords),
    #     sum(map(lambda c : c[1], mire_coords)) / len(mire_coords)
    # )
    # ratios = (
    #     mire_centroid[0]/osmnx_centroid[0],
    #     mire_centroid[1]/osmnx_centroid[1]
    # )
    # print("OSMNX centroid: {}".format(osmnx_centroid))
    # print("MIRE centroid: {}".format(mire_centroid))
    # print("ratios x: {}, y: {}".format(ratios[0], ratios[1]))

    # calculate max/min
    # osmnx_bounds_x = (
    #     min(map(lambda c : c[0], osmnx_coords)),
    #     max(map(lambda c : c[0], osmnx_coords))
    # )
    # osmnx_bounds_y = (
    #     min(map(lambda c : c[1], osmnx_coords)),
    #     max(map(lambda c : c[1], osmnx_coords))
    # )
    # # print("OSMNX bounds x: ", osmnx_bounds_x)
    # # print("OSMNX bounds y: ", osmnx_bounds_y)
    # osmnx_dimensions = (
    #         osmnx_bounds_x[1] - osmnx_bounds_x[0],
    #         osmnx_bounds_y[1] - osmnx_bounds_y[0]
    # )
    # # print("OSMNX dimensions: ", osmnx_dimensions)

    # mire_bounds_x = (
    #     min(map(lambda c : c[0], mire_coords)),
    #     max(map(lambda c : c[0], mire_coords))
    # )
    # mire_bounds_y = (
    #     min(map(lambda c : c[1], mire_coords)),
    #     max(map(lambda c : c[1], mire_coords))
    # )
    # # print("MIRE bounds x: ", mire_bounds_x)
    # # print("MIRE bounds y: ", mire_bounds_y)
    # mire_dimensions = (
    #     mire_bounds_x[1] - mire_bounds_x[0],
    #     mire_bounds_y[1] - mire_bounds_y[0]
    # )
    # print("MIRE dimensions: ", mire_dimensions)


    # adjust
    # x_scale = osmnx_dimensions[0] / mire_dimensions[0]
    # y_scale = osmnx_dimensions[1] / mire_dimensions[1]
    # x_scale = 0.01745329251994328
    # y_scale = 0.01745329251994328
    # x_trans = osmnx_bounds_x[0] - mire_bounds_x[0]*x_scale
    # y_trans = osmnx_bounds_y[1] - mire_bounds_y[1]*y_scale
    # x_trans = 0
    # y_trans = 0
    # mire_coords_adjusted = list(map(
    #     lambda c: (c[0] * x_scale + x_trans, c[1] * y_scale + y_trans),
    #     mire_coords
    # ))
    # print("xscale: {}\nyscale: {}\nxtrans: {}\nytrans: {}".format(x_scale, y_scale, x_trans, y_trans))

    # plot
    # chart = leather.Chart('MIRE coords (blue) vs osmnx coords (red)')
    # print(fiona.transform.transform('EPSG:4326', 'EPSG:26953', [-105.0], [40.0]))
    # fiona_xs, fiona_ys = fiona.transform.transform(
    #     'EPSG:26985', 'EPSG:4326',
    #     [c[0] for c in mire_coords],
    #     [c[1] for c in mire_coords]
    # )
    # fiona_translated_mire = list(zip(fiona_xs, fiona_ys))

    # ax = plt.subplot(111)
    # ax.scatter(fiona_xs, fiona_ys, color = 'c', s=16)

    # plt.show()

    # osmnx_xs = [k[0] for k in osmnx_coords]
    # osmnx_ys = [j[1] for j in osmnx_coords]


    # ax.scatter(osmnx_xs, osmnx_ys, color = 'r', s=4)

    # print(fiona_translated_mire[:10])

    # ax.set_facecolor((0.25,0.25,0.25))

    # plt.show()

    # chart.add_dots(fiona_translated_mire, name="MIRE", fill_color="#33dddd", radius=0.4)
    # chart.add_dots(osmnx_coords, name="OSMNX", fill_color="#dd3333", radius=0.2)
    # chart.add_dots([[180946.6307, 577713.4801], [25647.7745, 230853.3514]], fill_color="#000", radius=2.0)
    # chart.to_svg('../out/test.svg')

