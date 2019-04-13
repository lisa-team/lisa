import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# trip_id,lat,lon,time


trace_data_path = "/Volumes/SCOPE/SCOPE_Teams_2018-19/Volpe_Santos/data/2018-10_Bird_waypoints.csv"
with pd.read_csv(trace_data_path) as df:
    line = f.readline()
    print(line)

# x = np.arange(0, 5, 0.1)
# y = np.sin(x)
# plt.plot(x, y)
# plt.show()

