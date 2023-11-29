# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 14:14:06 2023


"""

import numpy as np
import scipy.stats as stats
from copy import deepcopy




def smooth(zones_vec):
    
    """
    This function can smooth the zones.
    
    :param zones_vec: array [n_cells, n_cells]  
    
    :return: smoothened mod_zones fields<np.array[n_cells_x, n_cells_y]>
        with zone distribution (ints from 1 to n_zones)
    """
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

    return mod_zones 


def hkfields(zones_vec, parameter_sets, n_zones):
    """
    Function assigns corresponding log(K) to each cells.

    :param zones_vec: array [n_cells, n_cells]
    
    :param parameter_values: array [n_mc, n_parameters]
        with parameter sets that need to be transformed into log(K) fields
    :param n_zones: <int>
        number of zones       
    :return: log(K) fields<np.array[n_cells_x, n_cells_y]>
        with zone distribution (ints from 1 to n_zones)
    
    """
    zones_vec = np.array(zones_vec, dtype=float)
    log_k_fields = zones_vec
    for i in range(n_zones):
        log_k_fields[np.where(log_k_fields == i+1)] = parameter_sets[0, i]

    return log_k_fields
