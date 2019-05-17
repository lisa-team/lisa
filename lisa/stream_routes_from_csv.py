import codecs
import csv
import os

import pickle


from pathlib import Path



def get_routes_from_single_csv(filename, debug = False):
    """
    Extract routes from a single csv.

    CSV should be formatted with each row separated by semicolons, as follows:
    
    route_id; start_time; end_time; lats; longs

    lats and longs should be chronologically-arranged latitude and longitude points.

    For a route in Washington DC, lats should be roughly 38.907192 and longs should be roughly -77.036873.

    """
    routes = []

    csv.field_size_limit(10**7)

    try:
        with open(filename, 'r') as fp:
            reader = csv.reader(fp, delimiter=';')

            # read CSV headers
            headers = next(reader)

            count = 0

            # read rest of file
            for row in reader:
                """
                row is length 5: route_id, start_time, end_time, lats, longs.

                lats and longs are a giant string with each x or y coordinate separated by commas.
                """
                # Print each row
                count+=1

                
                try:
                    row[3] = list(zip([float(k) for k in row[3].split(',')], [float(k) for k in row[4].split(',')]))

                    routes.append(row[3])

                except Exception as ex:
                    print("Get_routes_from_single_csv error 1:", ex, filename)
                    continue

                if debug:
                    
                    print(count)    
                
                    if count > 20:
                        break

    except Exception as ex:
        print("Get_routes_from_single_csv error 2:", ex, filename)

    return routes




if __name__ == "__main__":

    ENCODING = 'utf-8'



    PATH = "Z:\\SCOPE_Teams_2018-19\\Volpe_Santos\\data\\ddot\\processed\\"

    # testpath = Path('D:/test.txt')

    data_files = os.listdir(PATH)

    os.chdir("Z:\\")

    for data_file in data_files:
        filename = PATH + data_file

        if data_file.endswith(".csv"):
            pass

            
            # print(filename)

            routes = [] # this is the thing that gets pickled
            try:
                with open(filename, 'r') as fp:
                # with codecs.open(filename, "r", ENCODING) as fp:
                    reader = csv.reader(fp, delimiter=';')

                    # read CSV headers
                    headers = next(reader)
                    # print(headers)

                    count = 0

                    # read rest of file
                    for row in reader:
                        """
                        row is length 5: route_id, start_time, end_time, lats, longs.

                        lats and longs are a giant string with each x or y coordinate separated by commas.
                        """
                        # Print each row

                        if count < 20:
                            try:
                                row[3] = list(zip([float(k) for k in row[3].split(',')], [float(k) for k in row[4].split(',')]))
                                # row[4] = [float(k) for k in row[4].split(',')]

                                row = row[:4]

                                routes.append(row)

                            except Exception as ex:
                                print(1, ex, filename)
                                continue

                            count+=1
                        else:
                            break

            except:
                print(2, ex, filename)

        elif data_file.endswith(".p"):
            try:
                tmp = pickle.load(open(str(filename), "rb"))
                print(tmp[0])
            except Exception as ex:
                print(3, ex, filename)
                continue


