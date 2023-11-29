"""
Module contains the user input regarding the grounwater model setup.

Needed inputs:
-------------------
matlab_path : string
    full path where the matlab file to be run (MC_FEM_GW_main_python.m) is located.
true_run_path : string
    full path where the .mat file with the true, synthetic model run results can be found (needed if generate_true_run
    is False.
generate_true_run : bool
    True to generate a true, synthetic model run, False to read the previously generated data found in 'true_run_path'
n_reali : int
    number of zoned log(K) distributions to generate (number of Monte Carlo realizations to run at a time)

grid_size : list of floats
    With the number of cells in the [x, y] direction for the model grid, default: [50,50]
grid_distance : list of floats
    With the length in the [x,y] direction of each cell grid.
transport data : dictionary
    With the following transport data, which is kept constant for all model runs:
model_data : dictionary
    With data regarding the type of model (1: homogeneous, 2:zoned, 3:  geostatistical) and the number of zones, if
    model=2. For the zoned model, the number of zones corresponds to the number of uncertain parameters.

h_error : list of floats [2,1]
    With the error value associated to hydraulic head values (can be changed) in the first position, and the relative
    error in the second position, default is 0.
c_error : list of floats [2,1]
    With the error value associated to hydraulic head values (can be changed) in the first position, and the relative
    error in the second position. We consider the error for concentration depends on the measured value.

gm_name : string
    with the type of covariance used for the geostatistical model (default is "exponential")
gm_mean : float
    Mean log(K) value, used as the uncertain parameter prior's mean
gm_var : float
    Variance of log(K) values, used as the standard deviation for prior distribution (can be changed for a more
    'uninformative' prior
gm_correl_length : array [2,1]
    with the correlation length in [x, y] for the geostatistical model

"""
import math
import numpy as np

# Path where MATLAB code is found
matlab_path = r'C:/Users/tian/Desktop/initial_codes/matlab'  # Path where MATLAB executable is
matlab_data = f'C:/Users/tian/Desktop/initial_codes/matlab'
# Synthetic true run
true_run_path = f'C:/Users/tian/Desktop/initial_codes/matlab/OUTPUT/true_run.mat'
# k_true_path = os.path.join(matlab_path, "DATA", "saved_run_10", "Y_true.mat")

# GENERAL MODEL DATA ------------------------------------------------------------------------------------------------
generate_true_run = True  # True: generate a synthetic true run, False to read the previously generated true run

n_reali = 1  # Number of realizations to run

# Grid configuration: constant
grid_size = np.array([50.0, 50.0])  # Number of grid cells in the [x, y] direction
grid_distance = np.array([1.0, 1.0])  # Cell size in the [x, y] direction (m)

# Transport data ...................................................................................
# Constant transport data:
transport_data = {"porosity": 0.35,  # constant porosity
                  "long_disp": 2.5,  # constant longitudinal dispersivity
                  "transv_disp": 0.25,  # constant transversal dispersivity
                  "D": 1e-9  # constant diffusion coefficient
                  }

# Information for the model
model_data = {
    'model_type': 2,  # 1:Homogeneous, 2:Zoned, 3:Geostatistical
    'n_zones': 7  # if model_type = 2, else = 0
}

# Measurement data:
h_error = [0.06, 0]
c_error = [0.06, 0.2]

# ------------------------------------------------------------------------------------------------------------------ #
# Information for the geostatistical model
gm_name = "exponential"
gm_mean = math.log(1e-5)  # ln(K) average --> constant
gm_var = 1.0  # ln(K) variance --> constant
gm_correl_length = np.array([10.0, 10.0])  # correlation length in [x, y] for geostatistical model


# Generate geostatistical data dictionary
geostatistical_data = {"gm_name": gm_name,
                       "gm_mean": gm_mean,
                       "gm_var": gm_var,
                       "gm_correl_length": gm_correl_length}