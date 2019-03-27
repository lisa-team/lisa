import leather
import fiona
from fiona import transform
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
    temp = True
    with fiona.open("scratch_022819.gdb", 'r', layer=1) as inp:
        for f in inp:
            if temp:
                print(f)
                print(inp)
                temp = False
            mire_coords.append(f['geometry']['coordinates'])

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
    osmnx_bounds_x = (
        min(map(lambda c : c[0], osmnx_coords)),
        max(map(lambda c : c[0], osmnx_coords))
    )
    osmnx_bounds_y = (
        min(map(lambda c : c[1], osmnx_coords)),
        max(map(lambda c : c[1], osmnx_coords))
    )
    print("OSMNX bounds x: ", osmnx_bounds_x)
    print("OSMNX bounds y: ", osmnx_bounds_y)
    osmnx_dimensions = (
            osmnx_bounds_x[1] - osmnx_bounds_x[0],
            osmnx_bounds_y[1] - osmnx_bounds_y[0]
    )
    print("OSMNX dimensions: ", osmnx_dimensions)

    mire_bounds_x = (
        min(map(lambda c : c[0], mire_coords)),
        max(map(lambda c : c[0], mire_coords))
    )
    mire_bounds_y = (
        min(map(lambda c : c[1], mire_coords)),
        max(map(lambda c : c[1], mire_coords))
    )
    print("MIRE bounds x: ", mire_bounds_x)
    print("MIRE bounds y: ", mire_bounds_y)
    mire_dimensions = (
        mire_bounds_x[1] - mire_bounds_x[0],
        mire_bounds_y[1] - mire_bounds_y[0]
    )
    print("MIRE dimensions: ", mire_dimensions)


    # adjust
    # x_scale = osmnx_dimensions[0] / mire_dimensions[0]
    # y_scale = osmnx_dimensions[1] / mire_dimensions[1]
    x_scale = 0.01745329251994328
    y_scale = 0.01745329251994328
    # x_trans = osmnx_bounds_x[0] - mire_bounds_x[0]*x_scale
    # y_trans = osmnx_bounds_y[1] - mire_bounds_y[1]*y_scale
    x_trans = 0
    y_trans = 0
    mire_coords_adjusted = list(map(
        lambda c: (c[0] * x_scale + x_trans, c[1] * y_scale + y_trans),
        mire_coords
    ))
    print("xscale: {}\nyscale: {}\nxtrans: {}\nytrans: {}".format(x_scale, y_scale, x_trans, y_trans))

    # plot
    chart = leather.Chart('MIRE coords (blue) vs osmnx coords (red)')
    print(fiona.transform.transform('EPSG:4326', 'EPSG:26953', [-105.0], [40.0]))
    fiona_xs, fiona_ys = fiona.transform.transform(
        'EPSG:26985', 'EPSG:4326',
        [c[0] for c in mire_coords],
        [c[1] for c in mire_coords]
    )
    fiona_translated_mire = list(zip(fiona_xs, fiona_ys))
    chart.add_dots(fiona_translated_mire, name="MIRE", fill_color="#33dddd", radius=0.4)
    chart.add_dots(osmnx_coords, name="OSMNX", fill_color="#dd3333", radius=0.2)
    # chart.add_dots([[180946.6307, 577713.4801], [25647.7745, 230853.3514]], fill_color="#000", radius=2.0)
    chart.to_svg('../out/dc.svg')

