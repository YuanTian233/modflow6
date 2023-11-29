import numpy as np
import scipy.stats as stats
import math
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt


# Information for the geostatistical model
# n_zones = 7  # if model_type = 2, else = 0
# n_reali = 1
# gm_name = "exponential"
# gm_mean = math.log(1e-5)  # ln(K) average --> constant
# gm_var = 1.0  # ln(K) variance --> constant
# # correlation length in [x, y] for geostatistical model
# gm_correl_length = np.array([10.0, 10.0])
# parameter_sets = np.random.normal(loc=gm_mean,
#                                   scale=gm_var,
#                                   size=(n_reali, n_zones)
#                                   )





# fig = plt.figure(figsize=(6, 6))
# zones_vec01 = zones_vec81x41  # [0 : nrow, 0 : ncol]
# plt.imshow(zones_vec01)
# plt.title("Zones Field 81x41")
# plt.colorbar(shrink=0.8)


# fig = plt.figure(figsize=(6, 6))
# zones_vec02 = zones_vec50x50  # [0 : nrow, 0 : ncol]
# plt.imshow(zones_vec02)
# plt.title("Zones Field 50x50")
# plt.colorbar(shrink=0.8)



def smooth(zones_vec):
    mod_zones = deepcopy(zones_vec)

    for i in range(0, zones_vec.shape[0]):
        for j in range(0, zones_vec.shape[1]):
            # Get adjacent indexes
            n = len(mod_zones)-1
            m = len(mod_zones[0])-1

            # Initialising a vector array where adjacent elements will be stored
            v = []

            # Checking for adjacent elements and adding them to array
            # Deviation of row that gets adjusted according to the provided position

            #
            for dx in range(-1 if (i > 0) else 0, 2 if (i < n) else 1):
                # Deviation of the column that gets adjusted according to the provided position
                for dy in range(-1 if (j > 0) else 0, 2 if (j < m) else 1):
                    if dx != 0 or dy != 0:
                        v.append(mod_zones[i + dx][j + dy])
            #
            if np.count_nonzero(v == mod_zones[i, j]) < np.count_nonzero(v != mod_zones[i, j]):
                mod_zones[i, j] = stats.mode(v)[0]

    return mod_zones # 缩进没对齐到def


# fig = plt.figure(figsize=(6, 6))
# mod_zones01 = a#[0 : nrow, 0 : ncol]
# plt.imshow(mod_zones01)
# plt.title("Field smooth")
# plt.colorbar()
# plt.show()



def hkfields(zones_vec, parameter_sets):
    zones_vec = np.array(zones_vec, dtype=float)
    log_k_fields = zones_vec
    for i in range(7):
        log_k_fields[np.where(log_k_fields == i+1)] = parameter_sets[0, i]

    return log_k_fields # 缩进没对齐到def

# c  =  hkfields(zones_vec50x50, parameter_sets)
# fig = plt.figure(figsize=(6, 6))
# mod_zones01 = c#[0 : nrow, 0 : ncol]
# plt.imshow(mod_zones01)
# plt.title("Log(K) Field 50x50")
# plt.colorbar(shrink=0.8)
# plt.show()


# c  =  hkfields(zones_vec81x41, parameter_sets)
# fig = plt.figure(figsize=(6, 6))
# mod_zones01 = c#[0 : nrow, 0 : ncol]
# plt.imshow(mod_zones01)
# plt.title("Log(K) Field 81x41")
# plt.colorbar(shrink=0.4)
# plt.show()