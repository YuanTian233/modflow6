# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 14:43:22 2023

@author: tian
"""
import matplotlib
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import math
import scipy.stats as stats
from bayesian_inference import calculate_likelihood, calculate_likelihood_manual, compute_bme, bal_selection_criteria

import matplotlib.pyplot as plt

import arviz as az







model_output_path = os.getcwd()
if os.path.exists(model_output_path + '/plots'):
    print('path exists')
else:
    os.mkdir(model_output_path + '/plots')

model_output_path = os.getcwd()
# Extract results at observation locations
conc_obs = pd.read_csv((model_output_path + '/input/obs_concentration.csv'),header = 0, index_col = 0)
conc_obs= np.array(conc_obs)
n_obs= conc_obs.shape[1] 
h_obs = pd.read_csv((model_output_path + '/input/obs_waterhead.csv'),header = 0, index_col = 0)
h_obs = np.array(h_obs)
n_obs = h_obs.shape[1] 
from gwm import add_noise
# Measurement data:
h_error = [0.06, 0.01]
c_error = [0.06, 0.02]
conc_obs = add_noise(conc_obs, c_error,n_sizes = n_obs,seed=0,ret_error=True)[0]
conc_measurement_error = add_noise(conc_obs, c_error,n_sizes = n_obs, seed=0, ret_error=True)[1]
conc_measurement_error = conc_measurement_error[0, :]
h_obs = add_noise(h_obs, h_error,n_sizes = n_obs,seed=0,ret_error=True)[0]
h_measurement_error = add_noise(h_obs, h_error,n_sizes = n_obs, seed=0, ret_error=True)[1]
h_measurement_error = h_measurement_error[0, :]

obs = np.hstack((h_obs, conc_obs))
gm_mean = math.log(1e-5)  # ln(K) average --> constant
gm_var = 1.0  # ln(K) variance --> constant




# # prior
prior = pd.read_csv((model_output_path + '/ref_data/parameter_sets.csv'), dtype=np.float64, header = 0, index_col = 0)
prior = np.array(prior)
# Get prior probability for each independent parameter value in each set
prior_pdf_ind = stats.norm.pdf(prior, loc = gm_mean, scale = gm_var)
# Get prior probability for each set
prior_pdf_set = np.prod(prior_pdf_ind, axis=1)




conc_ref = pd.read_csv((model_output_path + '/ref_data/concentration.csv'), dtype=np.float64, header = 0, index_col = 0)
conc_ref_output = add_noise(conc_ref, c_error,n_sizes = n_obs,seed=0, ret_error=True)[0]
h_ref = pd.read_csv((model_output_path + '/ref_data/waterhead.csv'), dtype=np.float64, header = 0, index_col = 0)
h_ref_output = add_noise(h_ref, h_error,n_sizes = n_obs,seed=0, ret_error=True)[0]


ref = np.hstack((h_ref_output, conc_ref_output))


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
measurement_error = np.hstack((h_measurement_error,conc_measurement_error))
ref_bme, ref_re, ref_ie = compute_bme(model_predictions=ref, 
                                            observations=obs, 
                                            var=measurement_error**2,
                                            prior_pdf=prior_pdf_set, 
                                            prior_samples=prior)

tp_i = 30
iteration_limit = 370


# pre_mean = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_370/MH/post/IE/val_s_pre.csv'), dtype=np.float64, header = 0, index_col = 0)
# pre_mean = np.array(pre_mean)
# pre_mean = pre_mean.T


# pre_mean1 = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_370/MC/IE/val_s_pre.csv'), dtype=np.float64, header = 0, index_col = 0)
# pre_mean1 = np.array(pre_mean1)
# pre_mean1 = pre_mean1.T

# az.style.use("arviz-doc")
# fig= plt.subplots()
# data1 = np.array(pre_mean)[0:-1,4]
# data2 = np.array(ref)[0:370,4]  # Second dataset
# data3 = obs[:,4]  # Second dataset
# data4 = np.array(pre_mean1)[0:-1,4]
# # Create histograms for both datasets
# plt.plot(data1, label='BAL_post_MCMC', color='#0A65B6')
# plt.plot(data4, label='BAL_MC', color='#F43238')
# # plt.plot(data2, label='REF', color='#F43238')
# plt.axhline(y=data3, color='green', linestyle='--', label='OBS')

# plt.legend()
# # Customize the plot
# plt.title('Mean of H at obs location 5')
# plt.xlabel('Number of TPs')


# surrogate_prediction = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL50_300/comb_IE/surrogate_prediction.csv'), dtype=np.float64, header = 0, index_col = 0)
# h_RERE = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_370/MH/RE/h_RE.csv'), dtype=np.float64, header = 0, index_col = 0)
# c_RERE = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_370/MH/RE/c_RE.csv'), dtype=np.float64, header = 0, index_col = 0)

# h_IEIE = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_370/MH/IE/h_IE.csv'), dtype=np.float64, header = 0, index_col = 0)
# c_IEIE = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_370/MH/IE/c_IE.csv'), dtype=np.float64, header = 0, index_col = 0)

# CP_MH_RE = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_370/MH_RE/RE/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0)
# CP_MH_IE = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_370/MH_RE/IE/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0)


# h_RE = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_370/RE/h_RE.csv'), dtype=np.float64, header = 0, index_col = 0)
# c_RE = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_370/RE/c_RE.csv'), dtype=np.float64, header = 0, index_col = 0)

# CP_RE = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_370/RE/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0)
# CP_IE = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_370/IE/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0)

# h_IE = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_370/IE/h_IE.csv'), dtype=np.float64, header = 0, index_col = 0)
# c_IE = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_370/IE/c_IE.csv'), dtype=np.float64, header = 0, index_col = 0)


# zone_mean_mc = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_370/MC/IE/crit_val_s_pre.csv'), dtype=np.float64, header = 0, index_col = 0)
# zone_mean_mcmc = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_370/MH/IE/crit_val_s_pre.csv'), dtype=np.float64, header = 0, index_col = 0)   


# zone_1_mean_mc = np.array(zone_mean_mc)[0,:370]
# zone_3_mean_mc = np.array(zone_mean_mc)[4,:370]
# zone_1_mean_mcmc = np.array(zone_mean_mcmc)[0,:370]
# zone_3_mean_mcmc = np.array(zone_mean_mcmc)[4,:370]

# zone_1_cmean_mc = np.array(zone_mean_mc)[10,:370]
# zone_3_cmean_mc = np.array(zone_mean_mc)[12,:370]
# zone_1_cmean_mcmc = np.array(zone_mean_mcmc)[10,:370]
# zone_3_cmean_mcmc = np.array(zone_mean_mcmc)[12,:370]



# fig, axes = plt.subplots(2, 2)
# x_values = np.arange(zone_1_mean_mc.shape[0])
# axes[0,1].plot(x_values, zone_1_mean_mc, label='MC', color='#F43238')
# axes[0,1].plot(x_values, zone_1_mean_mcmc, label='MC', color='#0A65B6')
# axes[0,0].hist(zone_1_mean_mc, bins=20, alpha=0.8,label='MC', color='#F43238')
# axes[0,0].hist(zone_1_mean_mcmc, bins=20, alpha=0.5,label='MCMC', color='#0A65B6')
# axes[0,0].set_title('Location 1 Mean of Hydraulic head')
# axes[0,0].set_xlabel('Mean Hydraulic head')
# axes[0,0].legend()

# axes[1,1].plot(x_values, zone_3_mean_mc, label='MC', color='#F43238')
# axes[1,1].plot(x_values, zone_3_mean_mcmc, label='MC', color='#0A65B6')
# axes[1,0].hist(zone_3_mean_mc, bins=20, alpha=0.8,label='MC', color='#F43238')
# axes[1,0].hist(zone_3_mean_mcmc, bins=20, alpha=0.5,label='MCMC', color='#0A65B6')
# axes[1,0].set_title('Location 3 Mean of Hydraulic head')
# axes[1,0].set_xlabel('Mean Hydraulic head')
# axes[1,0].legend()

# plt.show()


# fig, axes = plt.subplots(4, 2, figsize=(8, 6))
# axes[0].hist(zone_1_cmean_mc, bins=20, alpha=0.8,label='MC', color='#F43238')
# axes[0].hist(zone_1_cmean_mcmc, bins=20, alpha=0.5,label='MCMC', color='#0A65B6')
# axes[0].set_title('Location 1 Mean of Concentration')
# axes[0].set_xlabel('Mean Concentration')
# axes[0].legend()
# axes[1].hist(zone_3_cmean_mc, bins=20, alpha=0.8,label='MC', color='#F43238')
# axes[1].hist(zone_3_cmean_mcmc, bins=20, alpha=0.5,label='MCMC', color='#0A65B6')
# axes[1].set_title('Location 3 Mean of Concentration')
# axes[1].set_xlabel('Mean Concentration')
# axes[1].legend()

# plt.show()




# # MC prior
h_mc_bme_re = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/BME/h_RE.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mc_bme_re = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/BME/c_RE.csv'), dtype=np.float64, header = 0, index_col = 0)
h_mc_bme_ie = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/BME/h_IE.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mc_bme_ie = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/BME/c_IE.csv'), dtype=np.float64, header = 0, index_col = 0)
h_mc_bme_bme = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/BME/h_BME.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mc_bme_bme = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/BME/c_BME.csv'), dtype=np.float64, header = 0, index_col = 0)

h_mc_re_re = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/RE/h_RE.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mc_re_re = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/RE/c_RE.csv'), dtype=np.float64, header = 0, index_col = 0)
h_mc_re_ie = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/RE/h_IE.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mc_re_ie = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/RE/c_IE.csv'), dtype=np.float64, header = 0, index_col = 0)
h_mc_re_bme = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/RE/h_BME.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mc_re_bme = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/RE/c_BME.csv'), dtype=np.float64, header = 0, index_col = 0)

h_mc_ie_re = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/IE/h_RE.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mc_ie_re = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/IE/c_RE.csv'), dtype=np.float64, header = 0, index_col = 0)
h_mc_ie_ie = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/IE/h_IE.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mc_ie_ie = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/IE/c_IE.csv'), dtype=np.float64, header = 0, index_col = 0)
h_mc_ie_bme = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/IE/h_BME.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mc_ie_bme = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/IE/c_BME.csv'), dtype=np.float64, header = 0, index_col = 0)


n_h_mc_bme_re = h_mc_bme_re/h_ref_re
n_c_mc_bme_re = c_mc_bme_re/c_ref_re
n_h_mc_bme_ie = h_mc_bme_ie/h_ref_ie
n_c_mc_bme_ie = c_mc_bme_ie/c_ref_ie
n_h_mc_bme_bme = h_mc_bme_bme/h_ref_bme
n_c_mc_bme_bme =c_mc_bme_bme/c_ref_bme

n_h_mc_re_re = h_mc_re_re/h_ref_re
n_c_mc_re_re = c_mc_re_re/c_ref_re
n_h_mc_re_ie = h_mc_re_ie/h_ref_ie
n_c_mc_re_ie = c_mc_re_ie/c_ref_ie
n_h_mc_re_bme = h_mc_re_bme/h_ref_bme
n_c_mc_re_bme =c_mc_re_bme/c_ref_bme

n_h_mc_ie_re = h_mc_ie_re/h_ref_re
n_c_mc_ie_re = c_mc_ie_re/c_ref_re
n_h_mc_ie_ie = h_mc_ie_ie/h_ref_ie
n_c_mc_ie_ie = c_mc_ie_ie/c_ref_ie
n_h_mc_ie_bme = h_mc_ie_bme/h_ref_bme
n_c_mc_ie_bme =c_mc_ie_bme/c_ref_bme


# MCMC prior
h_mh_bme_re = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/BME/h_RE.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mh_bme_re = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/BME/c_RE.csv'), dtype=np.float64, header = 0, index_col = 0)
h_mh_bme_ie = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/BME/h_IE.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mh_bme_ie = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/BME/c_IE.csv'), dtype=np.float64, header = 0, index_col = 0)
h_mh_bme_bme = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/BME/h_BME.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mh_bme_bme = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/BME/c_BME.csv'), dtype=np.float64, header = 0, index_col = 0)

h_mh_re_re = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/RE/h_RE.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mh_re_re = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/RE/c_RE.csv'), dtype=np.float64, header = 0, index_col = 0)
h_mh_re_ie = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/RE/h_IE.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mh_re_ie = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/RE/c_IE.csv'), dtype=np.float64, header = 0, index_col = 0)
h_mh_re_bme = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/RE/h_BME.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mh_re_bme = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/RE/c_BME.csv'), dtype=np.float64, header = 0, index_col = 0)

h_mh_ie_re = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/IE/h_RE.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mh_ie_re = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/IE/c_RE.csv'), dtype=np.float64, header = 0, index_col = 0)
h_mh_ie_ie = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/IE/h_IE.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mh_ie_ie = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/IE/c_IE.csv'), dtype=np.float64, header = 0, index_col = 0)
h_mh_ie_bme = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/IE/h_BME.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mh_ie_bme = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/IE/c_BME.csv'), dtype=np.float64, header = 0, index_col = 0)

n_h_mh_bme_re = h_mh_bme_re/h_ref_re
n_c_mh_bme_re = c_mh_bme_re/c_ref_re
n_h_mh_bme_ie = h_mh_bme_ie/h_ref_ie
n_c_mh_bme_ie = c_mh_bme_ie/c_ref_ie
n_h_mh_bme_bme = h_mh_bme_bme/h_ref_bme
n_c_mh_bme_bme =c_mh_bme_bme/c_ref_bme

n_h_mh_re_re = h_mh_re_re/h_ref_re
n_c_mh_re_re = c_mh_re_re/c_ref_re
n_h_mh_re_ie = h_mh_re_ie/h_ref_ie
n_c_mh_re_ie = c_mh_re_ie/c_ref_ie
n_h_mh_re_bme = h_mh_re_bme/h_ref_bme
n_c_mh_re_bme =c_mh_re_bme/c_ref_bme

n_h_mh_ie_re = h_mh_ie_re/h_ref_re
n_c_mh_ie_re = c_mh_ie_re/c_ref_re
n_h_mh_ie_ie = h_mh_ie_ie/h_ref_ie
n_c_mh_ie_ie = c_mh_ie_ie/c_ref_ie
n_h_mh_ie_bme = h_mh_ie_bme/h_ref_bme
n_c_mh_ie_bme =c_mh_ie_bme/c_ref_bme


# MC post
h_mc_post_bme_re = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/post/BME/h_RE.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mc_post_bme_re = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/post/BME/c_RE.csv'), dtype=np.float64, header = 0, index_col = 0)
h_mc_post_bme_ie = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/post/BME/h_IE.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mc_post_bme_ie = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/post/BME/c_IE.csv'), dtype=np.float64, header = 0, index_col = 0)
h_mc_post_bme_bme = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/post/BME/h_BME.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mc_post_bme_bme = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/post/BME/c_BME.csv'), dtype=np.float64, header = 0, index_col = 0)

h_mc_post_re_re = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/post/RE/h_RE.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mc_post_re_re = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/post/RE/c_RE.csv'), dtype=np.float64, header = 0, index_col = 0)
h_mc_post_re_ie = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/post/RE/h_IE.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mc_post_re_ie = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/post/RE/c_IE.csv'), dtype=np.float64, header = 0, index_col = 0)
h_mc_post_re_bme = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/post/RE/h_BME.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mc_post_re_bme = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/post/RE/c_BME.csv'), dtype=np.float64, header = 0, index_col = 0)


h_mc_post_ie_re = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/post/IE/h_RE.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mc_post_ie_re = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/post/IE/c_RE.csv'), dtype=np.float64, header = 0, index_col = 0)
h_mc_post_ie_ie = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/post/IE/h_IE.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mc_post_ie_ie = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/post/IE/c_IE.csv'), dtype=np.float64, header = 0, index_col = 0)
h_mc_post_ie_bme = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/post/IE/h_BME.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mc_post_ie_bme = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/post/IE/c_BME.csv'), dtype=np.float64, header = 0, index_col = 0)

n_h_mc_post_bme_re = h_mc_post_bme_re/h_ref_re
n_c_mc_post_bme_re = c_mc_post_bme_re/c_ref_re
n_h_mc_post_bme_ie = h_mc_post_bme_ie/h_ref_ie
n_c_mc_post_bme_ie = c_mc_post_bme_ie/c_ref_ie
n_h_mc_post_bme_bme = h_mc_post_bme_bme/h_ref_bme
n_c_mc_post_bme_bme =c_mc_post_bme_bme/c_ref_bme

n_h_mc_post_re_re = h_mc_post_re_re/h_ref_re
n_c_mc_post_re_re = c_mc_post_re_re/c_ref_re
n_h_mc_post_re_ie = h_mc_post_re_ie/h_ref_ie
n_c_mc_post_re_ie = c_mc_post_re_ie/c_ref_ie
n_h_mc_post_re_bme = h_mc_post_re_bme/h_ref_bme
n_c_mc_post_re_bme =c_mc_post_re_bme/c_ref_bme

n_h_mc_post_ie_re = h_mc_post_ie_re/h_ref_re
n_c_mc_post_ie_re = c_mc_post_ie_re/c_ref_re
n_h_mc_post_ie_ie = h_mc_post_ie_ie/h_ref_ie
n_c_mc_post_ie_ie = c_mc_post_ie_ie/c_ref_ie
n_h_mc_post_ie_bme = h_mc_post_ie_bme/h_ref_bme
n_c_mc_post_ie_bme =c_mc_post_ie_bme/c_ref_bme
# MH post
h_mh_post_bme_re = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/post/BME/h_RE.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mh_post_bme_re = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/post/BME/c_RE.csv'), dtype=np.float64, header = 0, index_col = 0)
h_mh_post_bme_ie = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/post/BME/h_IE.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mh_post_bme_ie = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/post/BME/c_IE.csv'), dtype=np.float64, header = 0, index_col = 0)
h_mh_post_bme_bme = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/post/BME/h_BME.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mh_post_bme_bme = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/post/BME/c_BME.csv'), dtype=np.float64, header = 0, index_col = 0)

h_mh_post_re_re = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/post/RE/h_RE.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mh_post_re_re = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/post/RE/c_RE.csv'), dtype=np.float64, header = 0, index_col = 0)
h_mh_post_re_ie = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/post/RE/h_IE.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mh_post_re_ie = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/post/RE/c_IE.csv'), dtype=np.float64, header = 0, index_col = 0)
h_mh_post_re_bme = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/post/RE/h_BME.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mh_post_re_bme = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/post/RE/c_BME.csv'), dtype=np.float64, header = 0, index_col = 0)


h_mh_post_ie_re = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/post/IE/h_RE.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mh_post_ie_re = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/post/IE/c_RE.csv'), dtype=np.float64, header = 0, index_col = 0)
h_mh_post_ie_ie = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/post/IE/h_IE.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mh_post_ie_ie = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/post/IE/c_IE.csv'), dtype=np.float64, header = 0, index_col = 0)
h_mh_post_ie_bme = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/post/IE/h_BME.csv'), dtype=np.float64, header = 0, index_col = 0)
c_mh_post_ie_bme = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/post/IE/c_BME.csv'), dtype=np.float64, header = 0, index_col = 0)

n_h_mh_post_bme_re = h_mh_post_bme_re/h_ref_re
n_c_mh_post_bme_re = c_mh_post_bme_re/c_ref_re
n_h_mh_post_bme_ie = h_mh_post_bme_ie/h_ref_ie
n_c_mh_post_bme_ie = c_mh_post_bme_ie/c_ref_ie
n_h_mh_post_bme_bme = h_mh_post_bme_bme/h_ref_bme
n_c_mh_post_bme_bme =c_mh_post_bme_bme/c_ref_bme

n_h_mh_post_re_re = h_mh_post_re_re/h_ref_re
n_c_mh_post_re_re = c_mh_post_re_re/c_ref_re
n_h_mh_post_re_ie = h_mh_post_re_ie/h_ref_ie
n_c_mh_post_re_ie = c_mh_post_re_ie/c_ref_ie
n_h_mh_post_re_bme = h_mh_post_re_bme/h_ref_bme
n_c_mh_post_re_bme =c_mh_post_re_bme/c_ref_bme

n_h_mh_post_ie_re = h_mh_post_ie_re/h_ref_re
n_c_mh_post_ie_re = c_mh_post_ie_re/c_ref_re
n_h_mh_post_ie_ie = h_mh_post_ie_ie/h_ref_ie
n_c_mh_post_ie_ie = c_mh_post_ie_ie/c_ref_ie
n_h_mh_post_ie_bme = h_mh_post_ie_bme/h_ref_bme
n_c_mh_post_ie_bme =c_mh_post_ie_bme/c_ref_bme


cre ='#61D786'
cie ='#61B2D7'
cbme ='#D76177'


color1='#61B2D7'
color2='#D76177'












"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
-------------Prior---MC----------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""



tname = f'Normalized Scores with Prior-based MC sampling'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4.944))
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)

ax1.set_title('Hydraulic-head', fontname='Arial', fontsize=14)
line1, = ax1.plot(n_h_mc_bme_bme, color=cbme, label=r'BME^{BAL}')  
line2, = ax1.plot(n_h_mc_re_bme, color=cre, label=r'$D_{KL}^{BAL}$')
line3, = ax1.plot(n_h_mc_ie_bme, color=cie, label=r'$H^{BAL}$')
ax1.legend(handles=[line1, line2, line3], prop={'size': 10})

# Set x and y labels for the first subplot
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel(r'$BME/BME_{Ref}$')

ax2.set_title('Concentration', fontname='Arial', fontsize=14)
line1, = ax2.plot(n_c_mc_bme_bme, color=cbme, label=r'$BME^{BAL}$')  
line2, = ax2.plot(n_c_mc_re_bme, color=cre, label=r'$D_{KL}^{BAL}$')
line3, = ax2.plot(n_c_mc_ie_bme, color=cie, label=r'$H^{BAL}$')
ax2.legend(handles=[line1, line2, line3], prop={'size': 10})

# Set x and y labels for the second subplot
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel(r'$BME/BME_{Ref}$')

# fig.suptitle(tname, fontname='Arial', fontsize=16)
# fig.subplots_adjust(top=0.9)

# Set font globally for all text elements
plt.rc('font', family='Arial')

filename = "figure_Normalized_prior_mc_bme.png"
plt.savefig((model_output_path + f'/plots/{filename}'))
plt.show()






tname = f'Normalized Scores with Prior-based MC sampling'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4.944))
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)

ax1.set_title('Hydraulic-head', fontname='Arial', fontsize=14)
line1, = ax1.plot(n_h_mc_bme_re, color=cbme, label=r'BME^{BAL}')  
line2, = ax1.plot(n_h_mc_re_re, color=cre, label=r'$D_{KL}^{BAL}$')
line3, = ax1.plot(n_h_mc_ie_re, color=cie, label=r'$H^{BAL}$')
ax1.legend(handles=[line1, line2, line3], prop={'size': 10})

# Set x and y labels for the first subplot
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel(r'$D_{KL}/D_{KL}^{REF}$')

ax2.set_title('Concentration', fontname='Arial', fontsize=14)
line1, = ax2.plot(n_c_mc_bme_re, color=cbme, label=r'$BME^{BAL}$')  
line2, = ax2.plot(n_c_mc_re_re, color=cre, label=r'$D_{KL}^{BAL}$')
line3, = ax2.plot(n_c_mc_ie_re, color=cie, label=r'$H^{BAL}$')
ax2.legend(handles=[line1, line2, line3], prop={'size': 10})

# Set x and y labels for the second subplot
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel(r'$D_{KL}/D_{KL}^{REF}$')

# fig.suptitle(tname, fontname='Arial', fontsize=16)
# fig.subplots_adjust(top=0.9)

# Set font globally for all text elements
plt.rc('font', family='Arial')

filename = "figure_Normalized_prior_mc_re.png"
plt.savefig((model_output_path + f'/plots/{filename}'))
plt.show()


tname = f'Normalized Scores with Prior-based MC sampling'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4.944))
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)

ax1.set_title('Hydraulic-head', fontname='Arial', fontsize=14)
line1, = ax1.plot(n_h_mc_bme_ie, color=cbme, label=r'BME^{BAL}')  
line2, = ax1.plot(n_h_mc_re_ie, color=cre, label=r'$D_{KL}^{BAL}$')
line3, = ax1.plot(n_h_mc_ie_ie, color=cie, label=r'$H^{BAL}$')
ax1.legend(handles=[line1, line2, line3], prop={'size': 10})

# Set x and y labels for the first subplot
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel(r'$H/H_{REF}$')

ax2.set_title('Concentration', fontname='Arial', fontsize=14)
line1, = ax2.plot(n_c_mc_bme_ie, color=cbme, label=r'$BME^{BAL}$')  
line2, = ax2.plot(n_c_mc_re_ie, color=cre, label=r'$D_{KL}^{BAL}$')
line3, = ax2.plot(n_c_mc_ie_ie, color=cie, label=r'$H^{BAL}$')
ax2.legend(handles=[line1, line2, line3], prop={'size': 10})

# Set x and y labels for the second subplot
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel(r'$H/H_{REF}$')

# fig.suptitle(tname, fontname='Arial', fontsize=16)
# fig.subplots_adjust(top=0.9)

# Set font globally for all text elements
plt.rc('font', family='Arial')

filename = "figure_Normalized_prior_mc_ie.png"
plt.savefig((model_output_path + f'/plots/{filename}'))
plt.show()






"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
-------------Post---MC----------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


tname = f'Normalized Scores with Post-based MC sampling'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4.944))
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)

ax1.set_title('Hydraulic-head', fontname='Arial', fontsize=14)
line1, = ax1.plot(n_h_mc_post_bme_bme, color=cbme, label=r'$BME^{BAL}$')  
line2, = ax1.plot(n_h_mc_post_re_bme, color=cre, label=r'$D_{KL}^{BAL}$')
line3, = ax1.plot(n_h_mc_post_ie_bme, color=cie, label=r'$H^{BAL}$')
ax1.legend(handles=[line1, line2, line3], prop={'size': 10})

# Set x and y labels for the first subplot
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel(r'$BME/BME_{Ref}$')

ax2.set_title('Concentration', fontname='Arial', fontsize=14)
line1, = ax2.plot(n_c_mc_post_bme_bme, color=cbme, label=r'$BME^{BAL}$')  
line2, = ax2.plot(n_c_mc_post_re_bme, color=cre, label=r'$D_{KL}^{BAL}$')
line3, = ax2.plot(n_c_mc_post_ie_bme, color=cie, label=r'$H^{BAL}$')
ax2.legend(handles=[line1, line2, line3], prop={'size': 10})

# Set x and y labels for the second subplot
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel(r'$BME/BME_{Ref}$')

# fig.suptitle(tname, fontname='Arial', fontsize=16)
# fig.subplots_adjust(top=0.9)

# Set font globally for all text elements
plt.rc('font', family='Arial')

filename = "figure_Normalized_post_mc_bme.png"
plt.savefig((model_output_path + f'/plots/{filename}'))
plt.show()








tname = f'Normalized Scores with Post-based MC sampling'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4.944))
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)

ax1.set_title('Hydraulic-head', fontname='Arial', fontsize=14)
line1, = ax1.plot(n_h_mc_post_bme_re, color=cbme, label=r'$BME^{BAL}$')  
line2, = ax1.plot(n_h_mc_post_re_re, color=cre, label=r'$D_{KL}^{BAL}$')
line3, = ax1.plot(n_h_mc_post_ie_re, color=cie, label=r'$H^{BAL}$')
ax1.legend(handles=[line1, line2, line3], prop={'size': 10})

# Set x and y labels for the first subplot
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel(r'$D_{KL}/D_{KL}^{REF}$')

ax2.set_title('Concentration', fontname='Arial', fontsize=14)
line1, = ax2.plot(n_c_mc_post_bme_re, color=cbme, label=r'$BME^{BAL}$')  
line2, = ax2.plot(n_c_mc_post_re_re, color=cre, label=r'$D_{KL}^{BAL}$')
line3, = ax2.plot(n_c_mc_post_ie_re, color=cie, label=r'$H^{BAL}$')
ax2.legend(handles=[line1, line2, line3], prop={'size': 10})

# Set x and y labels for the second subplot
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel(r'$D_{KL}/D_{KL}^{REF}$')

# fig.suptitle(tname, fontname='Arial', fontsize=16)
# fig.subplots_adjust(top=0.9)

# Set font globally for all text elements
plt.rc('font', family='Arial')

filename = "figure_Normalized_post_mc_re.png"
plt.savefig((model_output_path + f'/plots/{filename}'))
plt.show()




tname = f'Normalized Scores with Post-based MC sampling'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4.944))
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)

ax1.set_title('Hydraulic-head', fontname='Arial', fontsize=14)
line1, = ax1.plot(n_h_mc_post_bme_ie, color=cbme, label=r'$BME^{BAL}$')  
line2, = ax1.plot(n_h_mc_post_re_ie, color=cre, label=r'$D_{KL}^{BAL}$')
line3, = ax1.plot(n_h_mc_post_ie_ie, color=cie, label=r'$H^{BAL}$')
ax1.legend(handles=[line1, line2, line3], prop={'size': 10})

# Set x and y labels for the first subplot
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel(r'$H/H_{REF}$')

ax2.set_title('Concentration', fontname='Arial', fontsize=14)
line1, = ax2.plot(n_c_mc_post_bme_ie, color=cbme, label=r'$BME^{BAL}$')  
line2, = ax2.plot(n_c_mc_post_re_ie, color=cre, label=r'$D_{KL}^{BAL}$')
line3, = ax2.plot(n_c_mc_post_ie_ie, color=cie, label=r'$H^{BAL}$')
ax2.legend(handles=[line1, line2, line3], prop={'size': 10})

# Set x and y labels for the second subplot
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel(r'$H/H_{REF}$')

# fig.suptitle(tname, fontname='Arial', fontsize=16)
# fig.subplots_adjust(top=0.9)

# Set font globally for all text elements
plt.rc('font', family='Arial')

filename = "figure_Normalized_post_mc_ie.png"
plt.savefig((model_output_path + f'/plots/{filename}'))
plt.show()








"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
-------------Prior---MCMC----------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


tname = f'Normalized Scores with Prior-based MCMC sampling'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4.944))
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)

ax1.set_title('Hydraulic-head', fontname='Arial', fontsize=14)
line1, = ax1.plot(n_h_mh_bme_bme, color=cbme, label=r'$BME^{BAL}$')  
line2, = ax1.plot(n_h_mh_re_bme, color=cre, label=r'$D_{KL}^{BAL}$')
line3, = ax1.plot(n_h_mh_ie_bme, color=cie, label=r'$H^{BAL}$')
ax1.legend(handles=[line1, line2, line3], prop={'size': 10})

# Set x and y labels for the first subplot
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel(r'$BME/BME_{Ref}$')

ax2.set_title('Concentration', fontname='Arial', fontsize=14)
line1, = ax2.plot(n_c_mh_bme_bme, color=cbme, label=r'$BME^{BAL}$')  
line2, = ax2.plot(n_c_mh_re_bme, color=cre, label=r'$D_{KL}^{BAL}$')
line3, = ax2.plot(n_c_mh_ie_bme, color=cie, label=r'$H^{BAL}$')
ax2.legend(handles=[line1, line2, line3], prop={'size': 10})

# Set x and y labels for the second subplot
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel(r'$BME/BME_{Ref}$')

# fig.suptitle(tname, fontname='Arial', fontsize=16)
# fig.subplots_adjust(top=0.9)

# Set font globally for all text elements
plt.rc('font', family='Arial')

filename = "figure_Normalized_prior_mcmc_bme.png"
plt.savefig((model_output_path + f'/plots/{filename}'))
plt.show()

tname = f'Normalized Scores with Prior-based MCMC sampling'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4.944))
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)

ax1.set_title('Hydraulic-head', fontname='Arial', fontsize=14)
line1, = ax1.plot(n_h_mh_bme_re, color=cbme, label=r'$BME^{BAL}$')  
line2, = ax1.plot(n_h_mh_re_re, color=cre, label=r'$D_{KL}^{BAL}$')
line3, = ax1.plot(n_h_mh_ie_re, color=cie, label=r'$H^{BAL}$')
ax1.legend(handles=[line1, line2, line3], prop={'size': 10})

# Set x and y labels for the first subplot
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel(r'$D_{KL}/D_{KL}^{REF}$')

ax2.set_title('Concentration', fontname='Arial', fontsize=14)
line1, = ax2.plot(n_c_mh_bme_re, color=cbme, label=r'$BME^{BAL}$')  
line2, = ax2.plot(n_c_mh_re_re, color=cre, label=r'$D_{KL}^{BAL}$')
line3, = ax2.plot(n_c_mh_ie_re, color=cie, label=r'$H^{BAL}$')
ax2.legend(handles=[line1, line2, line3], prop={'size': 10})

# Set x and y labels for the second subplot
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel(r'$D_{KL}/D_{KL}^{REF}$')

# fig.suptitle(tname, fontname='Arial', fontsize=16)
# fig.subplots_adjust(top=0.9)

# Set font globally for all text elements
plt.rc('font', family='Arial')

filename = "figure_Normalized_prior_mcmc_re.png"
plt.savefig((model_output_path + f'/plots/{filename}'))
plt.show()

tname = f'Normalized Scores with Prior-based MCMC sampling'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4.944))
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)

ax1.set_title('Hydraulic-head', fontname='Arial', fontsize=14)
line1, = ax1.plot(n_h_mh_bme_ie, color=cbme, label=r'$BME^{BAL}$')  
line2, = ax1.plot(n_h_mh_re_ie, color=cre, label=r'$D_{KL}^{BAL}$')
line3, = ax1.plot(n_h_mh_ie_ie, color=cie, label=r'$H^{BAL}$')
ax1.legend(handles=[line1, line2, line3], prop={'size': 10})

# Set x and y labels for the first subplot
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel(r'$H/H_{REF}$')

ax2.set_title('Concentration', fontname='Arial', fontsize=14)
line1, = ax2.plot(n_c_mh_bme_ie, color=cbme, label=r'$BME^{BAL}$')  
line2, = ax2.plot(n_c_mh_re_ie, color=cre, label=r'$D_{KL}^{BAL}$')
line3, = ax2.plot(n_c_mh_ie_ie, color=cie, label=r'$H^{BAL}$')
ax2.legend(handles=[line1, line2, line3], prop={'size': 10})

# Set x and y labels for the second subplot
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel(r'$H/H_{REF}$')

# fig.suptitle(tname, fontname='Arial', fontsize=16)
# fig.subplots_adjust(top=0.9)

# Set font globally for all text elements
plt.rc('font', family='Arial')

filename = "figure_Normalized_prior_mcmc_ie.png"
plt.savefig((model_output_path + f'/plots/{filename}'))
plt.show()



"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
-------------Post---MCMC----------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


tname = f'Normalized Scores with Post-based MCMC sampling'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4.944))
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)

ax1.set_title('Hydraulic-head', fontname='Arial', fontsize=14)
line1, = ax1.plot(n_h_mh_post_bme_bme, color=cbme, label=r'$BME^{BAL}$')  
line2, = ax1.plot(n_h_mh_post_re_bme, color=cre, label=r'$D_{KL}^{BAL}$')
line3, = ax1.plot(n_h_mh_post_ie_bme, color=cie, label=r'$H^{BAL}$')
ax1.legend(handles=[line1, line2, line3], prop={'size': 10})

# Set x and y labels for the first subplot
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel(r'$BME/BME_{Ref}$')

ax2.set_title('Concentration', fontname='Arial', fontsize=14)
line1, = ax2.plot(n_c_mh_post_bme_bme, color=cbme, label=r'$BME^{BAL}$')  
line2, = ax2.plot(n_c_mh_post_re_bme, color=cre, label=r'$D_{KL}^{BAL}$')
line3, = ax2.plot(n_c_mh_post_ie_bme, color=cie, label=r'$H^{BAL}$')
ax2.legend(handles=[line1, line2, line3], prop={'size': 10})

# Set x and y labels for the second subplot
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel(r'$BME/BME_{Ref}$')

# fig.suptitle(tname, fontname='Arial', fontsize=16)
# fig.subplots_adjust(top=0.9)

# Set font globally for all text elements
plt.rc('font', family='Arial')

filename = "figure_Normalized_post_mcmc_bme.png"
plt.savefig((model_output_path + f'/plots/{filename}'))
plt.show()

tname = f'Normalized Scores with Post-based MCMC sampling'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4.944))
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)

ax1.set_title('Hydraulic-head', fontname='Arial', fontsize=14)
line1, = ax1.plot(n_h_mh_post_bme_re, color=cbme, label=r'$BME^{BAL}$')  
line2, = ax1.plot(n_h_mh_post_re_re, color=cre, label=r'$D_{KL}^{BAL}$')
line3, = ax1.plot(n_h_mh_post_ie_re, color=cie, label=r'$H^{BAL}$')
ax1.legend(handles=[line1, line2, line3], prop={'size': 10})

# Set x and y labels for the first subplot
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel(r'$D_{KL}/D_{KL}^{REF}$')

ax2.set_title('Concentration', fontname='Arial', fontsize=14)
line1, = ax2.plot(n_c_mh_post_bme_re, color=cbme, label=r'$BME^{BAL}$')  
line2, = ax2.plot(n_c_mh_post_re_re, color=cre, label=r'$D_{KL}^{BAL}$')
line3, = ax2.plot(n_c_mh_post_ie_re, color=cie, label=r'$H^{BAL}$')
ax2.legend(handles=[line1, line2, line3], prop={'size': 10})

# Set x and y labels for the second subplot
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel(r'$D_{KL}/D_{KL}^{REF}$')

# fig.suptitle(tname, fontname='Arial', fontsize=16)
# fig.subplots_adjust(top=0.9)

# Set font globally for all text elements
plt.rc('font', family='Arial')

filename = "figure_Normalized_post_mcmc_re.png"
plt.savefig((model_output_path + f'/plots/{filename}'))
plt.show()

tname = f'Normalized Scores with Post-based MCMC sampling'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4.944))
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)

ax1.set_title('Hydraulic-head', fontname='Arial', fontsize=14)
line1, = ax1.plot(n_h_mh_bme_ie, color=cbme, label=r'$BME^{BAL}$')  
line2, = ax1.plot(n_h_mh_re_ie, color=cre, label=r'$D_{KL}^{BAL}$')
line3, = ax1.plot(n_h_mh_ie_ie, color=cie, label=r'$H^{BAL}$')
ax1.legend(handles=[line1, line2, line3], prop={'size': 10})

# Set x and y labels for the first subplot
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel(r'$H/H_{REF}$')

ax2.set_title('Concentration', fontname='Arial', fontsize=14)
line1, = ax2.plot(n_c_mh_post_bme_ie, color=cbme, label=r'$BME^{BAL}$')  
line2, = ax2.plot(n_c_mh_post_re_ie, color=cre, label=r'$D_{KL}^{BAL}$')
line3, = ax2.plot(n_c_mh_post_ie_ie, color=cie, label=r'$H^{BAL}$')
ax2.legend(handles=[line1, line2, line3], prop={'size': 10})

# Set x and y labels for the second subplot
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel(r'$H/H_{REF}$')

# fig.suptitle(tname, fontname='Arial', fontsize=16)
# fig.subplots_adjust(top=0.9)

# Set font globally for all text elements
plt.rc('font', family='Arial')

filename = "figure_Normalized_post_mcmc_ie.png"
plt.savefig((model_output_path + f'/plots/{filename}'))
plt.show()




# tname = f'Normalized Scores with Posterior-based MC sampling'
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4.944))
# fig.tight_layout(pad=1.0)
# fig.subplots_adjust(hspace=0.25)
# ax1.set_title('Hydraulic-head', fontname='Arial', fontsize=14)
# line1, = ax1.plot(n_h_mc_post_bme_bme, color=cbme, label="BME")  
# line2, = ax1.plot(n_h_mc_post_bme_re, color=cre, label='RE')
# line3, = ax1.plot(n_h_mc_post_bme_ie, color=cie, label='IE')
# ax1.legend(handles=[line1, line2, line3], loc='lower right', prop={'size': 10})

# ax2.set_title('Concentration', fontname='Arial', fontsize=14)
# line1, = ax2.plot(n_c_mc_post_bme_bme, color=cbme, label='BME')  
# line2, = ax2.plot(n_c_mc_post_bme_re, color=cre, label='RE')
# line3, = ax2.plot(n_c_mc_post_bme_ie, color=cie, label='IE')
# ax2.legend(handles=[line1, line2, line3], loc='lower right', prop={'size': 10})

# fig.suptitle(tname, fontname='Arial', fontsize=16)
# fig.subplots_adjust(top=0.9)

# # Set font globally for all text elements
# matplotlib.rc('font', family='Arial')

# filename = "figure_Normalized_post_mc.png"
# plt.savefig((model_output_path + f'/plots/{filename}'))
# plt.show()



# tname = f'Normalized Scores with Posterior-based MCMC-MH sampling'
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4.944))
# fig.tight_layout(pad=1.0)
# fig.subplots_adjust(hspace=0.25)
# ax1.set_title('Hydraulic-head', fontname='Arial', fontsize=14)
# line1, = ax1.plot(n_h_mh_post_bme_bme, color=cbme, label="BME")  
# line2, = ax1.plot(n_h_mh_post_bme_re, color=cre, label='RE')
# line3, = ax1.plot(n_h_mh_post_bme_ie, color=cie, label='IE')
# ax1.legend(handles=[line1, line2, line3], loc='lower right', prop={'size': 10})

# ax2.set_title('Concentration', fontname='Arial', fontsize=14)
# line1, = ax2.plot(n_c_mh_post_bme_bme, color=cbme, label='BME')  
# line2, = ax2.plot(n_c_mh_post_bme_re, color=cre, label='RE')
# line3, = ax2.plot(n_c_mh_post_bme_ie, color=cie, label='IE')
# ax2.legend(handles=[line1, line2, line3], loc='lower right', prop={'size': 10})

# fig.suptitle(tname, fontname='Arial', fontsize=16)
# fig.subplots_adjust(top=0.9)

# # Set font globally for all text elements
# matplotlib.rc('font', family='Arial')

# filename = "figure_Normalized_post_mh.png"
# plt.savefig((model_output_path + f'/plots/{filename}'))
# plt.show()



tname = f'Normalized Scores with Prior-based sampling using BME Selection'
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 7.416))
fig.tight_layout(pad=3.0)
fig.subplots_adjust(hspace=0.5)

# Plot 1
ax1.set_title('RE: Hydraulic-head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax1.plot(n_h_mc_bme_re, color='#61B2D7', label=r'$MC^{BME}$', linewidth=1.5)  
line2, = ax1.plot(n_h_mh_bme_re, color='#D76177', label=r'$MCMC^{BME}$', linewidth=1.5)
ax1.legend(handles=[line1, line2], loc='lower right', prop={'size': 10})
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel(r'$D_{KL}/D_{KL}^{REF}$')

# Plot 2
ax2.set_title('IE: Hydraulic-head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax2.plot(n_h_mc_bme_ie, color='#61B2D7', label=r'$MC^{BME}$', linewidth=1.5)  
line2, = ax2.plot(n_h_mh_bme_ie, color='#D76177', label=r'$MCMC^{BME}$', linewidth=1.5)
ax2.legend(handles=[line1, line2], loc='upper right', prop={'size': 10})
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel(r'$H/H_{REF}$')

# Plot 3
ax3.set_title('BME: Hydraulic-head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax3.plot(n_h_mc_bme_bme, color='#61B2D7', label=r'$MC^{BME}$', linewidth=1.5)  
line2, = ax3.plot(n_h_mh_bme_bme, color='#D76177', label=r'$MCMC^{BME}$', linewidth=1.5)
ax3.legend(handles=[line1, line2], loc='lower right', prop={'size': 10})
ax3.set_xlabel('Number of model runs')
ax3.set_ylabel(r'$BME/BME_{Ref}$')


# Plot 4
ax4.set_title('RE: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax4.plot(n_c_mc_bme_re, color='#61B2D7', label=r'$MC^{BME}$', linewidth=1.5)  
line2, = ax4.plot(n_c_mh_bme_re, color='#D76177', label=r'$MCMC^{BME}$', linewidth=1.5)
ax4.legend(handles=[line1, line2], loc='lower right', prop={'size': 10})
ax4.set_xlabel('Number of model runs')
ax4.set_ylabel(r'$D_{KL}/D_{KL}^{REF}$')

# Plot 5
ax5.set_title('IE: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax5.plot(n_c_mc_bme_ie, color='#61B2D7', label=r'$MC^{BME}$', linewidth=1.5)  
line2, = ax5.plot(n_c_mh_bme_ie, color='#D76177', label=r'$MCMC^{BME}$', linewidth=1.5)
ax5.legend(handles=[line1, line2], loc='upper right', prop={'size': 10})
ax5.set_xlabel('Number of model runs')
ax5.set_ylabel(r'$H/H_{REF}$')

# Plot 6
ax6.set_title('BME: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax6.plot(n_c_mc_bme_bme, color='#61B2D7', label=r'$MC^{BME}$', linewidth=1.5)  
line2, = ax6.plot(n_c_mh_bme_bme, color='#D76177', label=r'$MCMC^{BME}$', linewidth=1.5)
ax6.legend(handles=[line1, line2], loc='lower right', prop={'size': 10})
ax6.set_xlabel('Number of model runs')
ax6.set_ylabel(r'$BME/BME_{Ref}$')

# fig.suptitle(tname, fontname='Arial', fontsize=16)
# fig.subplots_adjust(top=0.9)
filename = "figure_Normalized_bme_comp.png"
plt.savefig((model_output_path + f'/plots/{filename}'))
plt.show()


tname = f'Normalized Scores with Prior-based sampling using RE Selection'
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 7.416))
fig.tight_layout(pad=3.0)
fig.subplots_adjust(hspace=0.5)

# Plot 1
ax1.set_title('RE: Hydraulic-head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax1.plot(n_h_mc_re_re, color='#61B2D7', label=r'$MC^{D_{KL}}$', linewidth=1.5)  
line2, = ax1.plot(n_h_mh_re_re, color='#D76177', label=r'$MCMC^{D_{KL}}$', linewidth=1.5)
ax1.legend(handles=[line1, line2], loc='lower right', prop={'size': 10})
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel(r'$D_{KL}/D_{KL}^{REF}$')


# Plot 2
ax2.set_title('IE: Hydraulic-head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax2.plot(n_h_mc_re_ie, color='#61B2D7', label=r'$MC^{D_{KL}}$', linewidth=1.5)  
line2, = ax2.plot(n_h_mh_re_ie, color='#D76177', label=r'$MCMC^{D_{KL}}$', linewidth=1.5)
ax2.legend(handles=[line1, line2], loc='upper right', prop={'size': 10})
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel(r'$H/H_{REF}$')


# Plot 3
ax3.set_title('BME: Hydraulic-head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax3.plot(n_h_mc_re_bme, color='#61B2D7', label=r'$MC^{D_{KL}}$', linewidth=1.5)  
line2, = ax3.plot(n_h_mh_re_bme, color='#D76177', label=r'$MCMC^{D_{KL}}$', linewidth=1.5)
ax3.legend(handles=[line1, line2], loc='lower right', prop={'size': 10})
ax3.set_xlabel('Number of model runs')
ax3.set_ylabel(r'$BME/BME_{Ref}$')

# Plot 4
ax4.set_title('RE: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax4.plot(n_c_mc_re_re, color='#61B2D7', label=r'$MC^{D_{KL}}$', linewidth=1.5)  
line2, = ax4.plot(n_c_mh_re_re, color='#D76177', label=r'$MCMC^{D_{KL}}$', linewidth=1.5)
ax4.legend(handles=[line1, line2], loc='lower right', prop={'size': 10})
ax4.set_xlabel('Number of model runs')
ax4.set_ylabel(r'$D_{KL}/D_{KL}^{REF}$')

# Plot 5
ax5.set_title('IE: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax5.plot(n_c_mc_re_ie, color='#61B2D7', label=r'$MC^{D_{KL}}$', linewidth=1.5)  
line2, = ax5.plot(n_c_mh_re_ie, color='#D76177', label=r'$MCMC^{D_{KL}}$', linewidth=1.5)
ax5.legend(handles=[line1, line2], loc='upper right', prop={'size': 10})
ax5.set_xlabel('Number of model runs')
ax5.set_ylabel(r'$H/H_{REF}$')

# Plot 6
ax6.set_title('BME: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax6.plot(n_c_mc_re_bme, color='#61B2D7', label=r'$MC^{D_{KL}}$', linewidth=1.5)  
line2, = ax6.plot(n_c_mh_re_bme, color='#D76177', label=r'$MCMC^{D_{KL}}$', linewidth=1.5)
ax6.legend(handles=[line1, line2], loc='lower right', prop={'size': 10})
ax6.set_xlabel('Number of model runs')
ax6.set_ylabel(r'$BME/BME_{Ref}$')

# fig.suptitle(tname, fontname='Arial', fontsize=16)
# fig.subplots_adjust(top=0.9)
filename = "figure_Normalized_re_comp.png"
plt.savefig((model_output_path + f'/plots/{filename}'))
plt.show()


tname = f'Normalized Scores with Prior-based sampling using IE Selection'
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 7.416))
fig.tight_layout(pad=3.0)
fig.subplots_adjust(hspace=0.5)

# Plot 1
ax1.set_title('RE: Hydraulic-head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax1.plot(n_h_mc_ie_re, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax1.plot(n_h_mh_ie_re, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax1.legend(handles=[line1, line2], loc='lower right', prop={'size': 10})

# Plot 2
ax2.set_title('IE: Hydraulic-head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax2.plot(n_h_mc_ie_ie, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax2.plot(n_h_mh_ie_ie, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax2.legend(handles=[line1, line2], loc='upper right', prop={'size': 10})

# Plot 3
ax3.set_title('BME: Hydraulic-head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax3.plot(n_h_mc_ie_bme, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax3.plot(n_h_mh_ie_bme, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax3.legend(handles=[line1, line2], loc='lower right', prop={'size': 10})

# Plot 4
ax4.set_title('RE: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax4.plot(n_c_mc_ie_re, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax4.plot(n_c_mh_ie_re, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax4.legend(handles=[line1, line2], loc='lower right', prop={'size': 10})

# Plot 5
ax5.set_title('IE: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax5.plot(n_c_mc_ie_ie, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax5.plot(n_c_mh_ie_ie, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax5.legend(handles=[line1, line2], loc='upper right', prop={'size': 10})

# Plot 6
ax6.set_title('BME: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax6.plot(n_c_mc_ie_bme, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax6.plot(n_c_mh_ie_bme, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax6.legend(handles=[line1, line2], loc='lower right', prop={'size': 10})
fig.suptitle(tname, fontname='Arial', fontsize=16)
fig.subplots_adjust(top=0.9)
filename = "figure_Normalized_ie_comp.png"
plt.savefig((model_output_path + f'/plots/{filename}'))
plt.show()


tname = f'Normalized Scores with Posterior-based sampling using IE Selection'
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 7.416))
fig.tight_layout(pad=3.0)
fig.subplots_adjust(hspace=0.5)

# Plot 1
ax1.set_title('RE: Hydraulic-head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax1.plot(n_h_mc_post_ie_re, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax1.plot(n_h_mh_post_ie_re, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax1.legend(handles=[line1, line2], loc='lower right', prop={'size': 10})

# Plot 2
ax2.set_title('IE: Hydraulic-head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax2.plot(n_h_mc_post_ie_ie, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax2.plot(n_h_mh_post_ie_ie, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax2.legend(handles=[line1, line2], loc='upper right', prop={'size': 10})

# Plot 3
ax3.set_title('BME: Hydraulic-head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax3.plot(n_h_mc_post_ie_bme, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax3.plot(n_h_mh_post_ie_bme, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax3.legend(handles=[line1, line2], loc='lower right', prop={'size': 10})

# Plot 4
ax4.set_title('RE: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax4.plot(n_c_mc_post_ie_re, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax4.plot(n_c_mh_post_ie_re, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax4.legend(handles=[line1, line2], loc='lower right', prop={'size': 10})

# Plot 5
ax5.set_title('IE: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax5.plot(n_c_mc_post_ie_ie, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax5.plot(n_c_mh_post_ie_ie, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax5.legend(handles=[line1, line2], loc='upper right', prop={'size': 10})

# Plot 6
ax6.set_title('BME: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax6.plot(n_c_mc_post_ie_bme, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax6.plot(n_c_mh_post_ie_bme, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax6.legend(handles=[line1, line2], loc='lower right', prop={'size': 10})
fig.suptitle(tname, fontname='Arial', fontsize=16)
fig.subplots_adjust(top=0.9)
filename = "figure_Normalized_post_ie_comp.png"
plt.savefig((model_output_path + f'/plots/{filename}'))
plt.show()

tname = f'Normalized Scores with Posterior-based sampling using RE Selection'
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 7.416))
fig.tight_layout(pad=3.0)
fig.subplots_adjust(hspace=0.5)

# Plot 1
ax1.set_title('RE: Hydraulic-head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax1.plot(n_h_mc_post_re_re, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax1.plot(n_h_mh_post_re_re, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax1.legend(handles=[line1, line2], loc='lower right', prop={'size': 10})

# Plot 2
ax2.set_title('IE: Hydraulic-head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax2.plot(n_h_mc_post_re_ie, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax2.plot(n_h_mh_post_re_ie, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax2.legend(handles=[line1, line2], loc='upper right', prop={'size': 10})

# Plot 3
ax3.set_title('BME: Hydraulic-head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax3.plot(n_h_mc_post_re_bme, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax3.plot(n_h_mh_post_re_bme, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax3.legend(handles=[line1, line2], loc='lower right', prop={'size': 10})

# Plot 4
ax4.set_title('RE: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax4.plot(n_c_mc_post_re_re, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax4.plot(n_c_mh_post_re_re, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax4.legend(handles=[line1, line2], loc='lower right', prop={'size': 10})

# Plot 5
ax5.set_title('IE: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax5.plot(n_c_mc_post_re_ie, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax5.plot(n_c_mh_post_re_ie, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax5.legend(handles=[line1, line2], loc='upper right', prop={'size': 10})

# Plot 6
ax6.set_title('BME: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax6.plot(n_c_mc_post_re_bme, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax6.plot(n_c_mh_post_re_bme, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax6.legend(handles=[line1, line2], loc='lower right', prop={'size': 10})
fig.suptitle(tname, fontname='Arial', fontsize=16)
fig.subplots_adjust(top=0.9)
filename = "figure_Normalized_post_re_comp.png"
plt.savefig((model_output_path + f'/plots/{filename}'))
plt.show()

tname = f'Normalized Scores with Posterior-based sampling using BME Selection'
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(12, 7.416))
fig.tight_layout(pad=3.0)
fig.subplots_adjust(hspace=0.5)

# Plot 1
ax1.set_title('RE: Hydraulic-head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax1.plot(n_h_mc_post_bme_re, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax1.plot(n_h_mh_post_bme_re, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax1.legend(handles=[line1, line2], loc='lower right', prop={'size': 10})

# Plot 2
ax2.set_title('IE: Hydraulic-head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax2.plot(n_h_mc_post_bme_ie, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax2.plot(n_h_mh_post_bme_ie, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax2.legend(handles=[line1, line2], loc='upper right', prop={'size': 10})

# Plot 3
ax3.set_title('BME: Hydraulic-head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax3.plot(n_h_mc_post_bme_bme, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax3.plot(n_h_mh_post_bme_bme, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax3.legend(handles=[line1, line2], loc='lower right', prop={'size': 10})

# Plot 4
ax4.set_title('RE: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax4.plot(n_c_mc_post_bme_re, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax4.plot(n_c_mh_post_bme_re, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax4.legend(handles=[line1, line2], loc='lower right', prop={'size': 10})

# Plot 5
ax5.set_title('IE: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax5.plot(n_c_mc_post_bme_ie, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax5.plot(n_c_mh_post_bme_ie, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax5.legend(handles=[line1, line2], loc='upper right', prop={'size': 10})

# Plot 6
ax6.set_title('BME: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax6.plot(n_c_mc_post_bme_bme, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax6.plot(n_c_mh_post_bme_bme, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax6.legend(handles=[line1, line2], loc='lower right', prop={'size': 10})
fig.suptitle(tname, fontname='Arial', fontsize=16)
fig.subplots_adjust(top=0.9)
filename = "figure_Normalized_post_bme_comp.png"
plt.savefig((model_output_path + f'/plots/{filename}'))
plt.show()



cp_prior_mc_bme = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/BME/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]
cp_prior_mc_re = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/RE/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]
cp_prior_mc_ie = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/IE/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]


cp_prior_mh_bme = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/BME/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]
cp_prior_mh_re = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/RE/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]
cp_prior_mh_ie = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/IE/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]

tname = f'Training points for Zone 1 with Prior-based sampling using diffrent Selection'
fig, ((ax1, ax2, ax3)) = plt.subplots(3, 1, figsize=(12, 7.416))
fig.tight_layout(pad=3.0)
fig.subplots_adjust(hspace=0.5)

# Plot 1
ax1.set_title('RE', fontname='Arial', fontsize=12, weight='bold')
line1, = ax1.plot(cp_prior_mc_re, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax1.plot(cp_prior_mh_re, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax1.legend(handles=[line1, line2], loc='lower right', prop={'size': 9})

# Plot 2
ax2.set_title('IE', fontname='Arial', fontsize=12, weight='bold')
line1, = ax2.plot(cp_prior_mc_ie, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax2.plot(cp_prior_mh_ie, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax2.legend(handles=[line1, line2], loc='lower right', prop={'size': 9})

# Plot 3
ax3.set_title('BME', fontname='Arial', fontsize=12, weight='bold')
line1, = ax3.plot(cp_prior_mc_bme, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax3.plot(cp_prior_mh_bme, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax3.legend(handles=[line1, line2], loc='lower right', prop={'size': 9})


fig.suptitle(tname, fontname='Arial', fontsize=16)
fig.subplots_adjust(top=0.9)
filename = "figure_cp_prior_comp.png"
plt.savefig((model_output_path + f'/plots/{filename}'))
plt.show()

cp_prior_mc_bme = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/BME/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]
cp_prior_mc_re = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/RE/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]
cp_prior_mc_ie = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/IE/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]


cp_prior_mh_bme = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/BME/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]
cp_prior_mh_re = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/RE/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]
cp_prior_mh_ie = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/IE/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]


cp_post_mc_bme = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/post/BME/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]
cp_post_mc_re = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/post/RE/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]
cp_post_mc_ie = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/post/IE/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]


cp_post_mh_bme = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/post/BME/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]
cp_post_mh_re = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/post/RE/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]
cp_post_mh_ie = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/post/IE/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]



tname = f'Training points for Zone 1 with Posterior-based sampling using diffrent Selection'
fig, ((ax1, ax2, ax3)) = plt.subplots(3, 1, figsize=(12, 7.416))
fig.tight_layout(pad=3.0)
fig.subplots_adjust(hspace=0.5)

# Plot 1
ax1.set_title('RE', fontname='Arial', fontsize=12, weight='bold')
line1, = ax1.plot(cp_post_mc_re, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax1.plot(cp_post_mh_re, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax1.legend(handles=[line1, line2], loc='lower right', prop={'size': 9})

# Plot 2
ax2.set_title('IE', fontname='Arial', fontsize=12, weight='bold')
line1, = ax2.plot(cp_post_mc_ie, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax2.plot(cp_post_mh_ie, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax2.legend(handles=[line1, line2], loc='lower right', prop={'size': 9})

# Plot 3
ax3.set_title('BME', fontname='Arial', fontsize=12, weight='bold')
line1, = ax3.plot(cp_post_mc_bme, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax3.plot(cp_post_mh_bme, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax3.legend(handles=[line1, line2], loc='lower right', prop={'size': 9})


fig.suptitle(tname, fontname='Arial', fontsize=16)
fig.subplots_adjust(top=0.9)
filename = "figure_cp_post_comp.png"
plt.savefig((model_output_path + f'/plots/{filename}'))
plt.show()



tname = f'Training points for Zone 1 with Prior-based sampling using diffrent Selection'
fig, ((ax1, ax2, ax3)) = plt.subplots(3, 1, figsize=(12, 7.416))
fig.tight_layout(pad=3.0)
fig.subplots_adjust(hspace=0.5)

# Plot 1
ax1.set_title('RE', fontname='Arial', fontsize=12, weight='bold')
line1, = ax1.plot(cp_prior_mc_re, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax1.plot(cp_prior_mh_re, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax1.legend(handles=[line1, line2], loc='lower right', prop={'size': 9})

# Plot 2
ax2.set_title('IE', fontname='Arial', fontsize=12, weight='bold')
line1, = ax2.plot(cp_prior_mc_ie, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax2.plot(cp_prior_mh_ie, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax2.legend(handles=[line1, line2], loc='lower right', prop={'size': 9})

# Plot 3
ax3.set_title('BME', fontname='Arial', fontsize=12, weight='bold')
line1, = ax3.plot(cp_prior_mc_bme, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax3.plot(cp_prior_mh_bme, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax3.legend(handles=[line1, line2], loc='lower right', prop={'size': 9})


fig.suptitle(tname, fontname='Arial', fontsize=16)
fig.subplots_adjust(top=0.9)
filename = "figure_cp_prior_comp.png"
plt.savefig((model_output_path + f'/plots/{filename}'))
plt.show()






tname = f'Training points for Zone 1 sampling MC'
fig, ((ax1, ax2)) = plt.subplots(2, 1, figsize=(12, 7.416))
fig.tight_layout(pad=3.0)
fig.subplots_adjust(hspace=0.5)

# Plot 1
ax1.set_title('prior', fontname='Arial', fontsize=12, weight='bold')
line1, = ax1.plot(cp_prior_mc_re, color=cre, label='RE', linewidth=1.5)  
line2, = ax1.plot(cp_prior_mc_bme, color=cbme, label='BME', linewidth=1.5)
line3, = ax1.plot(cp_prior_mc_ie, color=cie, label='IE', linewidth=1.5)
ax1.legend(handles=[line1, line2, line3], prop={'size': 9})

# Plot 2
ax2.set_title('post', fontname='Arial', fontsize=12, weight='bold')
line1, = ax2.plot(cp_prior_mc_re, color=cre, label='RE', linewidth=1.5)  
line2, = ax2.plot(cp_prior_mc_bme, color=cbme, label='BME', linewidth=1.5)
line3, = ax1.plot(cp_prior_mc_ie, color=cie, label='IE', linewidth=1.5)
ax2.legend(handles=[line1, line2, line3], prop={'size': 9})

fig.suptitle(tname, fontname='Arial', fontsize=16)
fig.subplots_adjust(top=0.9)
filename = "figure_cp_prior_mc.png"
plt.savefig((model_output_path + f'/plots/{filename}'))
plt.show()



tname = f'Training points for Zone 1 sampling MH'
fig, ((ax1, ax2)) = plt.subplots(2, 1, figsize=(12, 7.416))
fig.tight_layout(pad=3.0)
fig.subplots_adjust(hspace=0.5)

# Plot 1
ax1.set_title('prior', fontname='Arial', fontsize=12, weight='bold')
line1, = ax1.plot(cp_prior_mh_re, color=cre, label='RE', linewidth=1.5)  
line2, = ax1.plot(cp_prior_mh_bme, color=cbme, label='BME', linewidth=1.5)
line3, = ax1.plot(cp_prior_mh_ie, color=cie, label='IE', linewidth=1.5)
ax1.legend(handles=[line1, line2, line3], prop={'size': 9})

# Plot 2
ax2.set_title('post', fontname='Arial', fontsize=12, weight='bold')
line1, = ax2.plot(cp_prior_mh_re, color=cre, label='RE', linewidth=1.5)  
line2, = ax2.plot(cp_prior_mh_bme, color=cbme, label='BME', linewidth=1.5)
line3, = ax1.plot(cp_prior_mh_ie, color=cie, label='IE', linewidth=1.5)
ax2.legend(handles=[line1, line2, line3], prop={'size': 9})

fig.suptitle(tname, fontname='Arial', fontsize=16)
fig.subplots_adjust(top=0.9)
filename = "figure_cp_prior_mh.png"
plt.savefig((model_output_path + f'/plots/{filename}'))
plt.show()


# fig= plt.subplots()
# data1 = np.array(CP_MH_RE)[0:-1,2]
# data2 = np.array(CP_RE)[0:-1,2]  # Second dataset
# # Create histograms for both datasets
# plt.hist(data1, bins=80, alpha=0.8, label='MCMC-BAL', color='#0A65B6')
# plt.hist(data2, bins=80, alpha=0.5, label='BAL', color='#F43238')
# plt.legend()
# # Customize the plot
# plt.title('Distribution of the Training Point 3 by RE')
# plt.xlabel('Hydraulic Conductivity')

# fig= plt.subplots()
# data1 = np.array(CP_MH_RE)[0:-1,1]
# data2 = np.array(CP_RE)[0:-1,1]  # Second dataset
# # Create histograms for both datasets
# plt.hist(data1, bins=80, alpha=0.8, label='MCMC-BAL', color='#0A65B6')
# plt.hist(data2, bins=80, alpha=0.5, label='BAL', color='#F43238')
# plt.legend()
# # Customize the plot
# plt.title('Distribution of the Training Point 2 by RE ')
# plt.xlabel('Hydraulic Conductivity')

# fig= plt.subplots()
# data1 = np.array(CP_MH_IE)[0:-1,2]
# data2 = np.array(CP_IE)[0:-1,2]  # Second dataset
# # Create histograms for both datasets
# plt.hist(data1, bins=80, alpha=0.8, label='MCMC-BAL', color='#0A65B6')
# plt.hist(data2, bins=80, alpha=0.5, label='BAL', color='#F43238')
# plt.legend()
# # Customize the plot
# plt.title('Distribution of the Training Point 3 by IE')
# plt.xlabel('Hydraulic Conductivity')

# fig= plt.subplots()
# data1 = np.array(CP_MH_IE)[0:-1,1]
# data2 = np.array(CP_IE)[0:-1,1]  # Second dataset
# # Create histograms for both datasets
# plt.hist(data1, bins=80, alpha=0.8, label='MCMC-BAL', color='#0A65B6')
# plt.hist(data2, bins=80, alpha=0.5, label='BAL', color='#F43238')
# plt.legend()
# # Customize the plot
# plt.title('Distribution of the Training Point 2 by IE')
# plt.xlabel('Hydraulic Conductivity')




#             # First dataset
#             # Second dataset


# fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(8, 12))
# data = np.array(CP_MH_RE)  
# colors = ['#F43299', '#F43238', '#3299F4', '#32F4EE', '#32F48D', '#88949E']
# # Plot each column separately
# for i in range(6):
#     axes[i].plot(data[:, i], color=colors[i])
#     axes[i].set_ylabel(f'Training Point {i+1}')

# # Add titles and labels
# axes[0].set_title('Training Points for MCMC BAL by RE')
# axes[-1].set_xlabel('Numbers of BAL Iteration')

# # Adjust layout
# plt.tight_layout()

# # Display the plot
# plt.show()


# fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(8, 12))
# data = np.array(CP_RE)  
# colors = ['#F43299', '#F43238', '#3299F4', '#32F4EE', '#32F48D', '#88949E']
# # Plot each column separately
# for i in range(6):
#     axes[i].plot(data[:, i], color=colors[i])
#     axes[i].set_ylabel(f'Training Point {i+1}')

# # Add titles and labels
# axes[0].set_title('Training Points for MCMC BAL by RE')
# axes[-1].set_xlabel('Numbers of BAL Iteration')

# # Adjust layout
# plt.tight_layout()

# # Display the plot
# plt.show()




# fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(8, 12))
# data = np.array(CP_MH_IE)  
# colors = ['#F43299', '#F43238', '#3299F4', '#32F4EE', '#32F48D', '#88949E']
# # Plot each column separately
# for i in range(6):
#     axes[i].plot(data[:, i], color=colors[i])
#     axes[i].set_ylabel(f'Training Point {i+1}')

# # Add titles and labels
# axes[0].set_title('Training Points for MCMC BAL by IE')
# axes[-1].set_xlabel('Numbers of BAL Iteration')

# # Adjust layout
# plt.tight_layout()

# # Display the plot
# plt.show()


# fig, axes = plt.subplots(nrows=6, ncols=1, figsize=(8, 12))
# data = np.array(CP_IE)  
# colors = ['#F43299', '#F43238', '#3299F4', '#32F4EE', '#32F48D', '#88949E']
# # Plot each column separately
# for i in range(6):
#     axes[i].plot(data[:, i], color=colors[i])
#     axes[i].set_ylabel(f'Training Point {i+1}')

# # Add titles and labels
# axes[0].set_title('Training Points for MCMC BAL by IE')
# axes[-1].set_xlabel('Numbers of BAL Iteration')

# # Adjust layout
# plt.tight_layout()

# # Display the plot
# plt.show()







# tname = f'Normalized RE for initial {tp_i} TP and {iteration_limit}  MCMC BAL TP'
# fig, (ax1,ax2)= plt.subplots(2,1)
# fig.tight_layout(pad=1.0)
# ax1.set_title('Hydraulic-head')
# ax1.plot(non_h, color='#61B2D7')
# ax2.set_title('Concentration')
# ax2.plot(non_c, color='#D76177')
# fig.suptitle(tname)
# fig.subplots_adjust(top=0.9) 
# filename = "figure_Normalized_BME.png" 
# plt.savefig((model_output_path+ f'/output/{filename}'))
# plt.show() 

# tname = f'Normalized IE for initial {tp_i} TP and {iteration_limit}  MCMC BAL TP'
# fig, (ax1,ax2)= plt.subplots(2,1)
# fig.tight_layout(pad=1.0)
# ax1.set_title('Hydraulic-head')
# ax1.plot(non_h2, color='#61B2D7')
# ax2.set_title('Concentration')
# ax2.plot(non_c2, color='#D76177')
# fig.suptitle(tname)
# fig.subplots_adjust(top=0.9) 
# filename = "figure_Normalized_BME.png" 
# plt.savefig((model_output_path+ f'/output/{filename}'))
# plt.show() 





# tname = f'Normalized IE for initial {tp_i} TP and {iteration_limit}  BAL TP'
# fig, (ax1,ax2)= plt.subplots(2,1)
# fig.tight_layout(pad=1.0)
# ax1.set_title('Hydraulic-head')
# ax1.plot(non_h11, color='#61B2D7')
# ax2.set_title('Concentration')
# ax2.plot(non_c11, color='#D76177')
# fig.suptitle(tname)
# fig.subplots_adjust(top=0.9) 
# filename = "figure_Normalized_BME.png" 
# plt.savefig((model_output_path+ f'/output/{filename}'))
# plt.show() 



# crti_MCMC = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_370/MH/post/IE/crit_val_IE.csv'), dtype=np.float64, header = 0, index_col = 0)

# crti_MC = pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_370/MC/IE/crit_val_IE.csv'), dtype=np.float64, header = 0, index_col = 0)

# fig= plt.subplots()
# data1 = np.array(crti_MC)
# data4 = np.array(crti_MCMC)  # Second dataset
# data3 = ref_ie
# # Create histograms for both datasets
# plt.hist(data1, bins=20, alpha=0.8,label='MC', color='#0A65B6')
# plt.hist(data4, bins=20, alpha=0.5,label='MCMC', color='#F43238')
# # plt.plot(data2, label='REF', color='#F43238')
# plt.axvline(x=data3, color='green', linestyle='--', label='REF')

# plt.legend()
# # Customize the plot
# plt.title('BAL Selection of IE ')
# plt.xlabel('Number of TPs')
# fig = plt.gcf()
# fig.set_size_inches(10, 6)







n_zones = 6 
n_reali =  1000


gm_mean = math.log(1e-5)  # ln(K) average --> constant
gm_var = 1.0  # ln(K) variance --> constant
parameter_sets = np.random.normal(loc=gm_mean, 
                                      scale=gm_var,
                                      size=(n_reali, n_zones)
                                      )

    
zones_vec50x50 = pd.read_csv((model_output_path + '/input/zone_distribution.csv'), dtype=np.float64)
zones_vec = np.array(zones_vec50x50, dtype = float)





point_loc = np.array(
    [(25, 25),
    (26, 31),
    (5, 18),
    (22, 7),
    (18, 27),
    (42, 36),
    (43, 10),
    (32, 20),
    (22, 32),
    (11, 41)]
    ) # this mark the location of obs points on plot


c  =  zones_vec
fig, (ax1)= plt.subplots()
cmap = plt.get_cmap('GnBu', 6)
mod_zones01 = c#[0 : nrow, 0 : ncol]
im3 = ax1.imshow(mod_zones01, cmap=cmap
                  #['#F309AC', '#F30937', '#09ACF3', '#F3C509', '#ACF309','#9703FF']) 
                  )
# ax1.set_title("Zone Distribution")

bounds = [0, 1 ,2, 3, 4, 5, 6]
cbar = fig.colorbar(im3, ax=ax1, ticks=bounds, spacing='proportional')

# Customize color bar ticks and labels
cbar.set_ticks([1.5, 2.25, 3.15, 4.0, 4.75, 5.65])
cbar.set_ticklabels(['Zone 1', 'Zone 2', 'Zone 3', 'Zone 4', 'Zone 5','Zone 6'])

# Add color bar label




ax1.scatter(point_loc[:, 1], point_loc[:, 0], marker='o',c='#112F2C', edgecolor = 'w')   
filename = "figure_zones.png"
plt.savefig((model_output_path + f'/plots/{filename}'))     
plt.show()        






from matplotlib.colors import ListedColormap

# Create a 50x50 grid
# grid_size = 50
# grid = np.zeros((grid_size, grid_size))  # Initialize the grid with zeros

# # Set the first column (column index 0) to red (value 1.0)
# grid[:, 0] = 1.0
# grid[:, -1] = 1.0
# grid[20:31, 0] = 0.5
# # Create a custom colormap with red for value 1.0 and white for value 0.0
# colors = ['white', 'red','blue']
# cmap = ListedColormap(colors)

# # Create a figure and axis for the map
# fig, ax = plt.subplots()

# # Display the grid as a map with the custom colormap
# im = ax.imshow(grid, cmap=cmap, interpolation='none', aspect='auto', extent=[0, grid_size, 0, grid_size])

# # Add a color bar for reference


# # Set axis labels and title
# ax.set_xlabel('No Flow')
# ax.set_ylabel('H = 1 m')
# ax.set_title('No Flow')
# ax_right = ax.twinx()
# ax_right.set_yticks(np.arange(0, grid_size+1, 10))
# ax_right.set_ylabel('H = 0 m')
# # Show the map
# plt.show()


# point_loc = np.array(
#     [(25, 25),
#     (26, 31),
#     (5, 18),
#     (22, 7),
#     (18, 27),
#     (42, 36),
#     (43, 10),
#     (32, 20),
#     (22, 32),
#     (11, 41)]
#     ) # this mark the location of obs points on plot


# # c  =  zones_vec
# # fig, (ax1)= plt.subplots()
# # cmap = plt.get_cmap('#09ACF3', 6)
# # mod_zones01 = c#[0 : nrow, 0 : ncol]
# # im3 = ax1.imshow(mod_zones01, cmap=cmap
# #                  #['#F309AC', '#F30937', '#09ACF3', '#F3C509', '#ACF309','#9703FF']) 
# #                  )
# # ax1.set_title("Zone Distribution")
# # bounds = [1, 2, 3, 4, 5, 6]
# # cbar = fig.colorbar(im3, ax=ax1,  ticks=bounds, spacing='proportional')



# # plt.show()  
# # Create a 50x50 grid
# grid_size = 50
# grid = np.zeros((grid_size, grid_size))  # Initialize the grid with zeros

# # Set the first column (column index 0) to red (value 1.0)
# grid[:, 0] = 1.0
# grid[:, -1] = 1.0
# grid[20:31, 0] = 0.5
# # Create a custom colormap with red for value 1.0 and white for value 0.0
# colors = ['white', '#CB1300','#00B8CB']
# cmap = ListedColormap(colors)

# # Create a figure and axis for the map
# fig, ax = plt.subplots(figsize=(5, 5))

# # Display the grid as a map with the custom colormap
# im = ax.imshow(grid, cmap=cmap, interpolation='none', aspect='auto', extent=[0, grid_size, 0, grid_size])

# # Add a color bar for reference


# # Set axis labels and title
# ax.scatter(point_loc[:, 1], point_loc[:, 0], marker='o',edgecolor = 'k',c='#3583CF')
# ax.set_xlabel('No Flow')
# ax.set_ylabel('H = 1 m')
# ax.set_title('No Flow')
# ax_right = ax.twinx()
# ax_right.set_yticks(np.arange(0, grid_size+1, 10))
# ax_right.set_ylabel('H = 0 m')
# # Show the map
# filename = "figure_50by50.png"
# plt.savefig((model_output_path + f'/plots/{filename}'))   
# plt.show()