import numpy as np
import scipy.stats as stats
import math
import pandas as pd
from scipy.io import loadmat
from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt





#grid = GRID(grid_size=np.array([50.0, 50.0]), grid_distance=np.array([1.0, 1.0]))


model_data = {
    'model_type': 2,  # 1:Homogeneous, 2:Zoned, 3:Geostatistical
    'n_zones': 7  # if model_type = 2, else = 0
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



# mat_true_data = loadmat('C:/Users/tian/Desktop/initial_codes/matlab/OUTPUT/true_run.mat')
# # Extract the true log(K) data:
# true_k_vector = mat_true_data['prior_k']

# data=loadmat('C:/Users/tian/Desktop/gwf_flopy_modflow6/input/hk.mat')
# true_k_vector = data['Y_true']



# y_max = math.ceil(np.max(true_k_vector))
# y_min = math.floor(np.min(true_k_vector))
#     # Zone distribution
# step = (y_max-y_min)/7
# zone_range = np.arange(start=y_min, stop=y_max+step, step=step)

# # Get zones based on true log(K) field
# original_zones = np.searchsorted(zone_range, true_k_vector)
# df = pd.DataFrame(original_zones)
# df.to_csv('original_zones_50x50.csv',index = False)

zones_vec  = pd.read_csv('C:/Users/tian/Desktop/test/original_zones_50x50.csv', dtype=np.float64)
# 1zones_vec = np.reshape(original_zones, (125, 100), order="F",)
zones_vec = np.array(zones_vec, dtype = float)




fig = plt.figure(figsize=(6, 6))
mod_zones01 = zones_vec#[0 : nrow, 0 : ncol]
im = plt.imshow(mod_zones01)
plt.title("Zones 02")
plt.colorbar(im)
plt.show()





for i in range(0, 7):
    zones_vec[zones_vec == i+1] = parameter_sets[0,i]
log_k_fields = zones_vec


# fig = plt.figure(figsize=(6, 6))
# mod_zones01 = log_k_fields#[0 : nrow, 0 : ncol]
# plt.imshow(mod_zones01)
# plt.title("Log(K) Field 02")
# plt.colorbar()
# plt.show()









