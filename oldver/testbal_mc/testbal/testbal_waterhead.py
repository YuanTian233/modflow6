# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 14:49:18 2023

@author: tian
"""
import numpy as np
import os
import shutil
import math
import pandas as pd
import scipy.stats as stats

import pickle

# import chaospy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import WhiteKernel

from bayesian_inference import calculate_likelihood, calculate_likelihood_manual, compute_bme, bal_selection_criteria
from plots import plot_size, plot_1d_gpe_final, plot_gpe_scores, plot_bal_criteria











# model_output_path = 'C:/Users/tian/Desktop/gp/testbal_mc'
model_output_path = os.getcwd()
if os.path.exists(model_output_path + '/output'):
    print('path exists')
else:
    os.mkdir(model_output_path + '/output')




model_name = "50"

n_zones = 6 





   
obs_hk = pd.read_csv((model_output_path + '/input/obs_hk.csv'), dtype=np.float64,header = 0, index_col = 0)
obs_hk = np.array(obs_hk, dtype = float)

    
# Extract results at observation locations
h_obs = pd.read_csv((model_output_path + '/input/obs_waterhead.csv'),header = 0, index_col = 0)
h_obs = np.array(h_obs)
n_obs = h_obs.shape[1] 
from testmc import add_noise
# Measurement data:
h_error = [0.06, 0.01]
# c_error = [0.06, 0.02]
h_obs = add_noise(h_obs, h_error,n_sizes = n_obs,seed=0,ret_error=True)[0]
h_measurement_error = add_noise(h_obs, h_error,n_sizes = n_obs, seed=0, ret_error=True)[1]
h_measurement_error = h_measurement_error[0, :]




# ------------------------------------------------------------------------------------------------------------------ #
# --------------------------------------- USER INPUT --------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


# Bayesian inference data:
mc_size = 10_000    # number of samples to get from the prior distribution
np.random.seed(0)
# Bayesian Active Learning data:
tp_i = 100
                # number of initial collocation points

iteration_limit = 2 # number of BAL iterations
d_size_AL = 1_000        # number of active learning sets (sets I take from the prior to do the active learning).
mc_size_AL = 1_000      # sample size for output space
# Note: d_size_AL+ iteration_limit < mc_size
al_strategy = "RE"   # BAL criteria (Here we want to test different methods)

# Gaussian Process data:
alpha = 0.0002                # Noise added to diagonal in GPE training
n_restarts = 10               # Number of optimizer starts
y_norm_gpe = True             # normalization for GPE training




gm_mean = math.log(1e-5)  # ln(K) average --> constant
gm_var = 1.0  # ln(K) variance --> constant


# # ------------------------------------------------------------------------------------------------------------------ #
# # --------------------------------------- Sample from prior -------------------------------------------------------- #
# # ------------------------------------------------------------------------------------------------------------------ #

# # prior
prior = pd.read_csv((model_output_path + '/ref_data/parameter_sets.csv'), dtype=np.float64, header = 0, index_col = 0)
prior = np.array(prior)
# Get prior probability for each independent parameter value in each set
prior_pdf_ind = stats.norm.pdf(prior, loc = gm_mean, scale = gm_var)
# Get prior probability for each set
prior_pdf_set = np.prod(prior_pdf_ind, axis=1)




from testmc import hkfields
from testmc import runmodelflow
exe_name = 'C:/Users/tian/Desktop/mf6.4.2/bin/mf6.exe'  #remamber to change this path.
zones_vec50x50 = pd.read_csv((model_output_path + '/input/zone_distribution.csv'), dtype=np.float64)
zones_vec = np.array(zones_vec50x50, dtype = float)
# # Get collocation points:
# collocation_points = stats.norm.rvs(loc = gm_mean, 
#                                         scale = gm_var,
#                                         size = (tp_i, n_zones)
#                                         )
collocation_points = pd.read_csv((os.getcwd() + r'/ref_data/parameter_sets.csv'), header = 0, index_col = 0)
collocation_points = np.array(collocation_points)
collocation_points = collocation_points[0:tp_i,]
# Extract results at observation locations
h_eva = pd.read_csv((os.getcwd() + r'/ref_data/waterhead.csv'),header = 0, index_col = 0)
h_eva = np.array(h_eva)[0:tp_i,]
h_eva = add_noise(h_eva, h_error,n_sizes = n_obs, seed=0, ret_error=True)[0]



# # ------------------------------------------------------------------------------------------------------------------ #
# # --------------------------------------- Get reference data-------------------------------------------------------- #
# # ------------------------------------------------------------------------------------------------------------------ #
# # Data to compare the surrogate model to
h_ref = pd.read_csv((model_output_path + '/ref_data/waterhead.csv'), dtype=np.float64, header = 0, index_col = 0)
# h_ref  = np.array(h_ref)[:,0:6]
h_ref_output = add_noise(h_ref, h_error,n_sizes = n_obs,seed=0, ret_error=True)[0]




# model_predictions=ref_output, observations=observations, var=measurement_error**2,
#                                       prior_pdf=prior_pdf_set, prior_samples=prior

# ref_output = nonlinear_model(t=t_, params=prior)
h_ref_bme, h_ref_re, h_ref_ie = compute_bme(model_predictions=h_ref_output, 
                                            observations=h_obs, 
                                            var=h_measurement_error**2,
                                            prior_pdf=prior_pdf_set, 
                                            prior_samples=prior
                                            )

# # # ------------------------------------------------------------------------------------------------------------------ #
# # # --------------------------------------- Train GPR -------------------------------------------------------- #
# # # ------------------------------------------------------------------------------------------------------------------ #



    
# Part 1. Initialization of information  ------------------------------------------------------------------------------
BME = np.zeros((iteration_limit+1, 1))
RE = np.zeros((iteration_limit+1, 1))
IE = np.zeros((iteration_limit+1, 1))
al_BME = np.zeros((d_size_AL, 1))
al_RE = np.zeros((d_size_AL, 1))
al_IE = np.zeros((d_size_AL, 1))
graph_list = []
graph_name = []
crit_val = np.zeros((iteration_limit, 1))


# # Loop for Bayesian update
for Ntp in range(0, iteration_limit+1):
    surrogate_prediction = np.zeros((n_obs,mc_size))
    surrogate_std = np.zeros((n_obs,mc_size))
    gp_list = []

    # Loop for each observation, which gets its own surrogate model
    for i, model in enumerate(h_eva.T):
        # 1. Set up kernel
        GP_kernel = (np.var(model) * RBF(length_scale=[1.0,1.0,1.0,1.0,1.0,1.0], 
                                         length_scale_bounds=[(1e-5, 1e5),
                                                              (1e-5, 1e5),
                                                              (1e-5, 1e5),
                                                              (1e-5, 1e5),
                                                              (1e-5, 1e5),
                                                              (1e-5, 1e5),
                                                              ]) 
                     + WhiteKernel(noise_level=0.06**2)
                     )
        # setup gp
        gp = GaussianProcessRegressor(kernel=GP_kernel, 
                                      alpha = alpha, 
                                      normalize_y=y_norm_gpe, 
                                      n_restarts_optimizer=n_restarts)
        # Train GP
        gp.fit(collocation_points, model)

        # Evaluate all prior parameter sets in GP
        surrogate_prediction[i, :], surrogate_std[i, :] = gp.predict(prior, return_std=True)

        # Save trained GP to use later
        gp_list.append(gp)
        # =============================================================================
        #         print(f"    Looping through loc {i}/{n_obs} --> hp: {gp.kernel_.theta}")
        # =============================================================================

#     # Computation of bayesian scores (in parameter space) -----------------------------------------------------
    total_error = (h_measurement_error ** 2) * np.ones(h_obs.shape[0])
    BME[Ntp], RE[Ntp], IE[Ntp] = compute_bme(surrogate_prediction.T, h_obs, total_error,
                                              prior_pdf=prior_pdf_set, prior_samples=prior)



         
    # Bayesian active learning (in output space) --------------------------------------------------------------
    if Ntp < iteration_limit:
        # Index of the elements of the prior distribution that have not been used as collocation points
        aux1 = np.where((prior[:d_size_AL + Ntp, :] == collocation_points[:, None]).all(-1))[1]
        aux2 = np.invert(np.in1d(np.arange(prior[:d_size_AL + Ntp, :].shape[0]), aux1))
        al_unique_index = np.arange(prior[:d_size_AL + Ntp, :].shape[0])[aux2]
         

        for iAL in range(0, len(al_unique_index)):
            # Exploration of output subspace associated with a defined prior combination.
            al_exploration = np.random.normal(size=(mc_size_AL, n_obs)) * surrogate_std[:, al_unique_index[iAL]] + \
                              surrogate_prediction[:, al_unique_index[iAL]]

            # BAL scores computation
            al_BME[iAL], al_RE[iAL] = compute_bme(model_predictions=al_exploration, observations=h_obs,
                                                  var=total_error)

        # Part 7. Selection criteria for next collocation point ------------------------------------------------------
        al_value, al_value_index = bal_selection_criteria(al_strategy, al_BME, al_RE)
        crit_val[Ntp, 0] = al_value

        # Part 8. Selection of new collocation point
        collocation_points = np.vstack((collocation_points, prior[al_unique_index[al_value_index], :]))

        # Part 9. Computation of the numerical model in the newly defined collocation point --------------------------
        parameter_sets = collocation_points[-1, :]
        parameter_sets = np.reshape(parameter_sets, (1, n_zones))
        k = hkfields(zones_vec = zones_vec, parameter_sets = parameter_sets, n_zones = n_zones)        
        k11 = np.exp(k)/100.0 * 86400  # cm^2/s -> m^2/d
        hk = k11
        sim_name = f'{model_name}_T'
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
        h_eva1 = np.array(h_eva1).reshape(1,n_obs)
        # h_eva1  = np.array(h_eva1)[:,0:6]
        h_eva1 = add_noise(h_eva1, h_error,n_sizes = n_obs, seed=0, ret_error=True)[0]

        
        h_eva = np.vstack((h_eva,h_eva1))

    # Progress report
    print("Bayesian iteration: " + str(Ntp + 1) + "/" + str(iteration_limit))

df = pd.DataFrame(collocation_points)        
df.to_csv(os.path.join(model_output_path+ r'/output','collocation_T_points.csv'))
df = pd.DataFrame(h_eva)        
df.to_csv(os.path.join(model_output_path+ r'/output','h_TP.csv'))
df = pd.DataFrame(surrogate_prediction)        
df.to_csv(os.path.join(model_output_path+ r'/output','surrogate_prediction.csv'))



    
file = open('gp.pickle', 'wb')
pickle.dump(gp,file)
file.close()
 
# Plot results:
# Get confidence intervals
lower_ci = surrogate_prediction - 2*surrogate_std
upper_ci = surrogate_prediction + 2*surrogate_std

if n_zones  == 1:
    plot_1d_gpe_final(prior, surrogate_prediction,  lower_ci, upper_ci, collocation_points,
                      h_eva, tp_i, iteration_limit, h_obs)



plot_gpe_scores(BME, RE, tp_i, ref_bme=h_ref_bme, ref_re=h_ref_re)
plot_bal_criteria(crit_val, al_strategy)

