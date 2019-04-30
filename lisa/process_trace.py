import sys
import os
import csv
import errno


def process_dockless(filename, destination):
    # route_id: {times: [], lats: [], lons: []}
    # route_id: [(time, lat, lon), ...]
    collector = {}

    # read csv
    with open(filename) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)  # throw out header

        for i, row in enumerate(reader):
            # value = collector.get(row[0], {"times": [], "lats": [], "lons": []})
            # value["times"].append(row[3])
            # value["lats"].append(row[1])
            # value["lons"].append(row[2])

            value = collector.get(row[0], [])
            value.append((row[3], row[1], row[2]))

            collector[row[0]] = value

    # sort on time
    for key in collector:
        collector[key] = sorted(collector.get(key))

    # make sure the path exists
    # if not os.path.exists(os.path.dirname(filename)):
    #     try:
    #         os.makedirs(os.path.dirname(filename))
    #     except OSError as exc:  # Guard against race condition
    #         if exc.errno != errno.EEXIST:
    #             raise

    # write to csv
    with open(destination, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=';')
        for key in collector:
            route_data = collector[key]
            # TODO: fix the times
            min_time_iter = (d[0] for d in route_data)
            max_time_iter = (d[0] for d in route_data)
            lat_iter = (d[1] for d in route_data)
            lon_iter = (d[2] for d in route_data)
            row = [
                key,
                min(min_time_iter) + "bad", max(max_time_iter) + "bad",
                ",".join(lat_iter), ",".join(lon_iter)
            ]
            # row = [
            #     key,
            #     min(data["times"]) + "bad", max(data["times"]) + "bad",
            #     ",".join(data["lats"]), ",".join(data["lons"])
            # ]
            writer.writerow(row)


if __name__ == "__main__":
    process_dockless(sys.argv[1], sys.argv[2])
