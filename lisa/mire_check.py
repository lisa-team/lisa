import leather
import fiona
import os

from graph import Graph, Name


if __name__ == "__main__":
    filename = "WashingtonDCGraph.pkl"

    # get osmnx node coords
    if not os.path.exists(filename):
        name = Name("Washington DC")
        g = Graph(name)
        g.save(filename)
    else:
        g = Graph.from_file(filename)
    osmnx_coords = [(a['x'], a['y']) for n, a in g.init_graph.nodes(data=True)]

    # get mire node coords
    mire_coords = []
    with fiona.open("scratch_022819.gdb", 'r', layer=1) as inp:
        for f in inp:
            mire_coords.append(f['geometry']['coordinates'])

    # calculate averages
    osmnx_centroid = (
        sum(map(lambda c : c[0], osmnx_coords)) / len(osmnx_coords),
        sum(map(lambda c : c[1], osmnx_coords)) / len(osmnx_coords)
    )
    mire_centroid = (
        sum(map(lambda c : c[0], mire_coords)) / len(mire_coords),
        sum(map(lambda c : c[1], mire_coords)) / len(mire_coords)
    )
    ratios = (
        mire_centroid[0]/osmnx_centroid[0],
        mire_centroid[1]/osmnx_centroid[1]
    )
    print("OSMNX centroid: {}".format(osmnx_centroid))
    print("MIRE centroid: {}".format(mire_centroid))
    print("ratios x: {}, y: {}".format(ratios[0], ratios[1]))

    # adjust
    mire_coords_adjusted = list(map(
        lambda c : (c[0]/3600 -180, c[1]/3600),
        mire_coords
    ))

    print(type(mire_coords))
    print(type(mire_coords_adjusted))

    # plot
    chart = leather.Chart('MIRE coords (blue) vs osmnx coords (red)')
    chart.add_dots(mire_coords_adjusted, name="MIRE", fill_color="#33dddd")
    chart.add_dots(osmnx_coords, name="OSMNX", fill_color="#dd3333")
    chart.to_svg('../out/dc.svg')

