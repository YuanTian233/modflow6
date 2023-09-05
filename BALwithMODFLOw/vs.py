# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 14:43:22 2023

@author: tian
"""

import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
import scipy.stats as stats
from bayesian_inference import calculate_likelihood, calculate_likelihood_manual, compute_bme, bal_selection_criteria



model_output_path = os.getcwd()
# Extract results at observation locations
conc_obs = pd.read_csv((model_output_path + '/input/obs_concentration.csv'),header = 0, index_col = 0)
conc_obs= np.array(conc_obs)
n_obs= conc_obs.shape[1] 
h_obs = pd.read_csv((model_output_path + '/input/obs_waterhead.csv'),header = 0, index_col = 0)
h_obs = np.array(h_obs)
n_obs = h_obs.shape[1] 
from testmc import add_noise
# Measurement data:
h_error = [0.06, 0.01]
c_error = [0.06, 0.02]
conc_obs = add_noise(conc_obs, c_error,n_sizes = n_obs,seed=0,ret_error=True)[0]
conc_measurement_error = add_noise(conc_obs, c_error,n_sizes = n_obs, seed=0, ret_error=True)[1]
conc_measurement_error = conc_measurement_error[0, :]
h_obs = add_noise(h_obs, h_error,n_sizes = n_obs,seed=0,ret_error=True)[0]
h_measurement_error = add_noise(h_obs, h_error,n_sizes = n_obs, seed=0, ret_error=True)[1]
h_measurement_error = h_measurement_error[0, :]

tp_i = 100
gm_mean = math.log(1e-5)  # ln(K) average --> constant
gm_var = 1.0  # ln(K) variance --> constant




# # prior
prior = pd.read_csv((model_output_path + '/ref_data/parameter_sets.csv'), dtype=np.float64, header = 0, index_col = 0)
prior = np.array(prior)
# Get prior probability for each independent parameter value in each set
prior_pdf_ind = stats.norm.pdf(prior, loc = gm_mean, scale = gm_var)
# Get prior probability for each set
prior_pdf_set = np.prod(prior_pdf_ind, axis=1)


collocation_points = pd.read_csv((os.getcwd() + r'/ref_data/parameter_sets.csv'), header = 0, index_col = 0)
collocation_points = np.array(collocation_points)
collocation_points = collocation_points[0:tp_i,]
# Extract results at observation locations
conc_eva = pd.read_csv((os.getcwd() + r'/ref_data/concentration.csv'),header = 0, index_col = 0)
conc_eva = np.array(conc_eva)[0:tp_i,]
conc_eva = add_noise(conc_eva, c_error,n_sizes = n_obs, seed=0, ret_error=True)[0]
h_eva = pd.read_csv((os.getcwd() + r'/ref_data/waterhead.csv'),header = 0, index_col = 0)
h_eva = np.array(h_eva)[0:tp_i,]
h_eva = add_noise(h_eva, h_error,n_sizes = n_obs, seed=0, ret_error=True)[0]

conc_ref = pd.read_csv((model_output_path + '/ref_data/concentration.csv'), dtype=np.float64, header = 0, index_col = 0)
conc_ref_output = add_noise(conc_ref, c_error,n_sizes = n_obs,seed=0, ret_error=True)[0]
h_ref = pd.read_csv((model_output_path + '/ref_data/waterhead.csv'), dtype=np.float64, header = 0, index_col = 0)
h_ref_output = add_noise(h_ref, h_error,n_sizes = n_obs,seed=0, ret_error=True)[0]





c_ref_bme, c_ref_re, c_ref_ie = compute_bme(
                                            model_predictions=conc_ref_output, 
                                            observations=conc_obs, 
                                            var=conc_measurement_error**2,
                                            prior_pdf=prior_pdf_set,
                                            prior_samples=prior)

h_ref_bme, h_ref_re, h_ref_ie = compute_bme(model_predictions=h_ref_output, 
                                            observations=h_obs, 
                                            var=h_measurement_error**2,
                                            prior_pdf=prior_pdf_set, 
                                            prior_samples=prior)










                                           
iteration_limit = 300
d_size_AL = 1000                                            
h_BME = np.zeros((iteration_limit+1, 1))
h_RE = np.zeros((iteration_limit+1, 1))
h_IE = np.zeros((iteration_limit+1, 1))

c_BME = np.zeros((iteration_limit+1, 1))
c_RE = np.zeros((iteration_limit+1, 1))
c_IE = np.zeros((iteration_limit+1, 1))

h1_BME = np.zeros((iteration_limit+1, 1))
h1_RE = np.zeros((iteration_limit+1, 1))
h1_IE = np.zeros((iteration_limit+1, 1))

c1_BME = np.zeros((iteration_limit+1, 1))
c1_RE = np.zeros((iteration_limit+1, 1))
c1_IE = np.zeros((iteration_limit+1, 1))

al_BME = np.zeros((d_size_AL, 1))
al_RE = np.zeros((d_size_AL, 1))
al_IE = np.zeros((d_size_AL, 1))


surrogate_prediction = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL50_300/comb_IE/surrogate_prediction.csv'), dtype=np.float64, header = 0, index_col = 0)
surrogate_prediction = np.array(surrogate_prediction.T)


surrogate_prediction_h1 = surrogate_prediction[0:10000,0:10]

surrogate_prediction_c1 = surrogate_prediction[0:10000,10:20]

surrogate_prediction_h = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL100_400/waterhead/surrogate_prediction.csv'), dtype=np.float64, header = 0, index_col = 0)
surrogate_prediction_h = np.array(surrogate_prediction_h.T)
surrogate_prediction_c = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL100_400/conc/surrogate_prediction.csv'), dtype=np.float64, header = 0, index_col = 0)
surrogate_prediction_c = np.array(surrogate_prediction_c.T)
# 
for i in range(0,iteration_limit+1):
    total_error_h = (h_measurement_error ** 2) * np.ones(h_obs.shape[0])
    h_BME[i], h_RE[i], h_IE[i] = compute_bme(surrogate_prediction_h[0:i+1,], 
                                        h_obs, 
                                        total_error_h,
                                        prior_pdf=prior_pdf_set, 
                                        prior_samples=prior)
    total_error_c = (conc_measurement_error ** 2) * np.ones(h_obs.shape[0])
    c_BME[i], c_RE[i], c_IE[i] = compute_bme(surrogate_prediction_c[0:i+1,], 
                                             conc_obs, 
                                             total_error_c,
                                             prior_pdf=prior_pdf_set, 
                                             prior_samples=prior)


    h1_BME[i], h1_RE[i], h1_IE[i] = compute_bme(surrogate_prediction_h1[0:i+1,], 
                                            h_obs, 
                                            total_error_h,
                                            prior_pdf=prior_pdf_set, 
                                            prior_samples=prior)

    c1_BME[i], c1_RE[i], c1_IE[i] = compute_bme(surrogate_prediction_c1[0:i+1,], 
                                                 conc_obs, 
                                                 total_error_c,
                                                 prior_pdf=prior_pdf_set, 
                                                 prior_samples=prior)


    
# for i in range(0,2):
#     total_error = (h_measurement_error ** 2) * np.ones(h_obs.shape[0])
#     BME[i], RE[i], IE[i] = compute_bme(surrogate_prediction[0,i:], h_obs, total_error,
#                                           prior_pdf=prior_pdf_set, prior_samples=prior)

# for Ntp in range(0, iteration_limit+1):
#     total_error = (h_measurement_error ** 2) * np.ones(h_obs.shape[0])
#     BME[Ntp], RE[Ntp], IE[Ntp] = compute_bme(surrogate_prediction, h_obs, total_error,
#                                           prior_pdf=prior_pdf_set, prior_samples=prior)



import matplotlib.pyplot as plt

import arviz as az


non_h = h_BME / h_ref_bme
non_h1 =h1_BME / h_ref_bme
non_c = c_BME / c_ref_bme
non_c1 =c1_BME / c_ref_bme
# data = {'concentration':non_c,
#          'waterhead':non_h
#          }



# fig, ((ax1,ax2),(ax3,ax4))= plt.subplots(2,2)
fig, (ax3,ax4)= plt.subplots(2,1)
fig.tight_layout(pad=1.0)
fig.suptitle("Normalized BME for initial 100 TP and 400 BAL TP")
fig.subplots_adjust(top=0.9) 
# ax1.set_title('Waterhead separate')
# ax1.plot(non_h)
# ax2.set_title('Concentration separate')
# ax2.plot(non_c)
ax3.set_title('Waterhead comb')
ax3.plot(non_h1)
ax4.set_title('Concentration combe')
ax4.plot(non_c1)

# ax[0].grid(True)
# ax = az.plot_trace(data,combined=True)

# plt.set_title("Normalized BME for initial 100 TP and 400 BAL TP")
# fig.suptitle("Normalized BME for initial 100 TP and 400 BAL TP")
plt.show()
