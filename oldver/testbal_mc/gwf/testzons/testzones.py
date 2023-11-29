import numpy as np
import scipy.stats as stats
import math
import pandas as pd
from scipy.io import loadmat
from copy import deepcopy
import matplotlib.pyplot as plt
from gw_model_functions import GRID
from gw_model_functions import generate_zoned_k
from gw_model_functions import generate_zoned_model





grid = GRID(grid_size=np.array([50.0, 50.0]), grid_distance=np.array([1.0, 1.0]))


model_data = {
    'model_type': 2,  # 1:Homogeneous, 2:Zoned, 3:Geostatistical
    'n_zones': 6  # if model_type = 2, else = 0
}
# Information for the geostatistical model
n_reali =  1
gm_name = "exponential"
gm_mean = math.log(1e-5)  # ln(K) average --> constant
gm_var = 1.0  # ln(K) variance --> constant
gm_correl_length = np.array([10.0, 10.0])  # correlation length in [x, y] for geostatistical model
parameter_sets = np.random.normal(loc=gm_mean, scale=gm_var,
                                  size=(n_reali, model_data['n_zones']))
a = parameter_sets[0,0]
b = parameter_sets[0,1]



# mat_true_data = loadmat('C:/Users/tian/Desktop/testzones/Y_true.mat')
# # Extract the true log(K) data:
# true_k_vector = mat_true_data['Y_true']

# # Considering a normal prior log(K), sample n_reali paramter sets:
# parameter_sets = np.random.normal(loc=gm_mean, scale=gm_var,
#                                   size=(n_reali, 7))

# # Generate informed zone distribution (based on true log(K) field)
# zone_distribution = generate_zoned_model(true_k_vector, 7, grid)

# df = pd.DataFrame(zone_distribution)

# df.to_csv('zone_distribution.csv', index=False)

zones_vec = pd.read_csv('C:/Users/tian/Desktop/testzones/zone_distribution.csv', dtype=np.float64)
zones_vec1 = np.array(zones_vec, dtype = float)






