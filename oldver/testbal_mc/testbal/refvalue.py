# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 19:08:53 2023

@author: tian
"""
import numpy as np
import os
import shutil
import math
import pandas as pd
import scipy.stats as stats
import time as time
import pickle
import arviz as az
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import WhiteKernel

from bayesian_inference import calculate_likelihood, calculate_likelihood_manual, compute_bme, bal_selection_criteria
from plots import plot_size, plot_1d_gpe_final, plot_gpe_scores, plot_bal_criteria

from gwm import hkfields
from gwm import runmodelflow

# ------------------------------------------------------------------------------------------------------------------ #
# --------------------------------------- USER INPUT --------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #
T = time.time()
# model_output_path = 'C:/Users/tian/Desktop/gp/testbal_mc'
outputfolder = 'ref'
model_output_path = os.getcwd()
if os.path.exists(os.path.join(model_output_path, outputfolder)):
    print('Path exists')
else:
    os.mkdir(os.path.join(model_output_path, outputfolder))

from gwm import add_noise
# Measurement data:
h_error = [0.06, 0.01]
c_error = [0.06, 0.02]

np.random.seed(0)


model_name = "00"

n_zones = 6 

exe_name = 'C:/Users/tian/Desktop/mf6.4.2/bin/mf6.exe'  #remamber to change this path.
zones_vec50x50 = pd.read_csv((model_output_path + '/input/zone_distribution.csv'), dtype=np.float64)
zones_vec = np.array(zones_vec50x50, dtype = float)
# # Get collocation points:
# collocation_points = stats.norm.rvs(loc = gm_mean, 
#                                         scale = gm_var,
#                                         size = (tp_i, n_zones)
#                                         )

a = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/post/IE/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))
a1 = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/post/IE/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1]
for i in range(0,270):
    parameter_sets = a1[i, :]
    parameter_sets = np.reshape(parameter_sets, (1, n_zones))
    k = hkfields(zones_vec = zones_vec, parameter_sets = parameter_sets, n_zones = n_zones)
    mat_file_path = (model_output_path + '/ref_data/Y_true.mat')
    k11 = np.exp(k)/100.0 * 86400  # cm^2/s -> m^2/d
    hk = k11
    sim_name = f'{model_name}'
    ws = os.path.join('model',sim_name) #worksapce for each model
    gwfname = "f" + sim_name # a name you can change for groundwater flow model. Name maximum length of 16
    gwtname = "t" + sim_name # a name you can change for groundwater transport model. Name maximum length of 16 
    runmodelflow(sim_name = sim_name, 
                  sim_ws = ws,
                  exe_name = exe_name,
                  gwfname = gwfname,
                  gwtname = gwtname,
                  hk = hk,
                  plot = False) 
    h_eva1 = pd.read_csv((os.getcwd() + f'/model/{sim_name}/{gwfname}.obs.head.csv'), dtype=np.float64, header = 0, usecols=range(1,11),skiprows=1)[-1:]
    h_eva1 = np.array(h_eva1).reshape(1,10)
    # h_eva1  = np.array(h_eva1)[:,0:6]
    h_eva1 = add_noise(h_eva1, h_error,n_sizes = 10, seed=0, ret_error=True)[0]
    
    
    
    h_eva = np.vstack((h_eva1,h_eva1))
    
    conc_eva1 = pd.read_csv((os.getcwd() + f'/model/{sim_name}/{gwtname}.obs.conc.csv'), dtype=np.float64, header = 0, usecols=range(1,11),skiprows=1)[-1:]
    conc_eva1 = np.array(conc_eva1).reshape(1,10)
    # h_eva1  = np.array(h_eva1)[:,0:6]
    conc_eva1 = add_noise(conc_eva1, c_error,n_sizes = 10, seed=0, ret_error=True)[0]
    
    
    conc_eva = np.vstack((conc_eva1,conc_eva1))
    eva = np.hstack((h_eva, conc_eva))    
    
    
    
    eva1 = np.hstack((h_eva1, conc_eva1))
    
    
    data_to_save = [
        (pd.DataFrame(eva1), 'eva.csv'),
        (pd.DataFrame(eva), 'comb_TP.csv'),
        (pd.DataFrame(surrogate_prediction), 'surrogate_prediction.csv'),
        # (pd.DataFrame(al_size), 'al_size.csv'),
        (pd.DataFrame(h_BME), 'h_BME.csv'),
        (pd.DataFrame(c_BME), 'c_BME.csv'),
        (pd.DataFrame(h_RE), 'h_RE.csv'),
        (pd.DataFrame(c_RE), 'c_RE.csv'),
        (pd.DataFrame(h_IE), 'h_IE.csv'),
        (pd.DataFrame(c_IE), 'c_IE.csv'),
        (pd.DataFrame(crit_val_BME), 'crit_val_BME.csv'),
        (pd.DataFrame(crit_val_RE), 'crit_val_RE.csv'),
        (pd.DataFrame(crit_val_IE), 'crit_val_IE.csv'),
        (pd.DataFrame(crit_val_s_std), 'crit_val_s_std.csv'),
        (pd.DataFrame(crit_val_s_pre), 'crit_val_s_pre.csv'),
        (pd.DataFrame(val_s_std), 'val_s_std.csv'),
        (pd.DataFrame(val_s_pre), 'val_s_pre.csv')
    ]

    # Save each dataframe to a CSV file
    for df, file_name in data_to_save:
        df.to_csv(os.path.join(model_output_path, outputfolder, file_name))