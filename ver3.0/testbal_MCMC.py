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
import matplotlib.pyplot as plt
import pickle
import arviz as az
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import WhiteKernel

from bayesian_inference import calculate_likelihood, calculate_likelihood_manual, compute_bme, bal_selection_criteria
from plots import plot_size, plot_1d_gpe_final, plot_gpe_scores, plot_bal_criteria
import time as time




# ------------------------------------------------------------------------------------------------------------------ #
# --------------------------------------- USER INPUT --------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #
T = time.time()
# model_output_path = 'C:/Users/tian/Desktop/gp/testbal_mc'
model_output_path = os.getcwd()
if os.path.exists(model_output_path + '/output'):
    print('path exists')
else:
    os.mkdir(model_output_path + '/output')




model_name = "50"

n_zones = 6 

# Bayesian inference data:
mc_size = 10_000    # number of samples to get from the prior distribution

# Bayesian Active Learning data:
tp_i = 30
                # number of initial collocation points
MCMC = 1000
iteration_limit = 270 # number of BAL iterations
d_size_AL = 1_000        # number of active learning sets (sets I take from the prior to do the active learning).
mc_size_AL = 1_000      # sample size for output space
# Note: d_size_AL+ iteration_limit < mc_size
al_strategy = "IE"   # BAL criteria (Here we want to test different methods)
# BME 6.6676338954766585 h
# RE  6.747033240397771 h
# IE  6.884736772245831 h

# Gaussian Process data:
alpha = 0.0002                # Noise added to diagonal in GPE training
n_restarts = 10               # Number of optimizer starts
y_norm_gpe = True             # normalization for GPE training




gm_mean = math.log(1e-5)  # ln(K) average --> constant
gm_var = 1.0  # ln(K) variance --> constant


# # ------------------------------------------------------------------------------------------------------------------ #
# # --------------------------------------- Sample from prior -------------------------------------------------------- #
# # ------------------------------------------------------------------------------------------------------------------ #

# prior
prior_ref = pd.read_csv((model_output_path + '/ref_data/parameter_sets.csv'), dtype=np.float64, header = 0, index_col = 0)
prior_ref = np.array(prior_ref)
# Get prior probability for each independent parameter value in each set
prior_pdf_ind_ref = stats.norm.pdf(prior_ref, loc = gm_mean, scale = gm_var)
# Get prior probability for each set
prior_pdf_set_ref = np.prod(prior_pdf_ind_ref, axis=1)

# # ------------------------------------------------------------------------------------------------------------------ #
# # --------------------------------------- obs -------------------------------------------------------- #
# # ------------------------------------------------------------------------------------------------------------------ #   
obs_hk = pd.read_csv((model_output_path + '/input/obs_hk.csv'), dtype=np.float64,header = 0, index_col = 0)
obs_hk = np.array(obs_hk, dtype = float)

    
# Extract results at observation locations
h_obs = pd.read_csv((model_output_path + '/input/obs_waterhead.csv'),header = 0, index_col = 0)
h_obs = np.array(h_obs)
 

conc_obs = pd.read_csv((model_output_path + '/input/obs_concentration.csv'),header = 0, index_col = 0)
conc_obs= np.array(conc_obs)


from gwm import add_noise
# Measurement data:
h_error = [0.06, 0.01]
c_error = [0.06, 0.02]
h_obs = add_noise(h_obs, h_error,n_sizes = 10,seed=0,ret_error=True)[0]
h_measurement_error = add_noise(h_obs, h_error,n_sizes = 10, seed=0, ret_error=True)[1]
h_measurement_error = h_measurement_error[0, :]

conc_obs = add_noise(conc_obs, c_error,n_sizes = 10,seed=0,ret_error=True)[0]
conc_measurement_error = add_noise(conc_obs, c_error,n_sizes = 10, seed=0, ret_error=True)[1]
conc_measurement_error = conc_measurement_error[0, :]

measurement_error = np.hstack((h_measurement_error,conc_measurement_error))

obs = np.hstack((h_obs, conc_obs))
n_obs= obs.shape[1]
       
        


from gwm import hkfields
from gwm import runmodelflow
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
h_eva = add_noise(h_eva, h_error,n_sizes = 10, seed=0, ret_error=True)[0]

conc_eva = pd.read_csv((os.getcwd() + r'/ref_data/concentration.csv'),header = 0, index_col = 0)
conc_eva = np.array(conc_eva)[0:tp_i,]
conc_eva = add_noise(conc_eva, c_error,n_sizes = 10, seed=0, ret_error=True)[0]


eva = np.hstack((h_eva,conc_eva))
# # ------------------------------------------------------------------------------------------------------------------ #
# # --------------------------------------- Get reference data-------------------------------------------------------- #
# # ------------------------------------------------------------------------------------------------------------------ #
# # Data to compare the surrogate model to
h_ref = pd.read_csv((model_output_path + '/ref_data/waterhead.csv'), dtype=np.float64, header = 0, index_col = 0)
# h_ref  = np.array(h_ref)[:,0:6]
h_ref_output = add_noise(h_ref, h_error,n_sizes = 10,seed=0, ret_error=True)[0]

conc_ref = pd.read_csv((model_output_path + '/ref_data/concentration.csv'), dtype=np.float64, header = 0, index_col = 0)
# h_ref  = np.array(h_ref)[:,0:6]
conc_ref_output = add_noise(conc_ref, c_error,n_sizes = 10,seed=0, ret_error=True)[0]


ref = np.hstack((h_ref_output, conc_ref_output))
ref_bme, ref_re, ref_ie = compute_bme(model_predictions = ref, 
                                      observations = obs, 
                                      var = measurement_error**2,
                                      prior_pdf = prior_pdf_set_ref, 
                                      prior_samples = prior_pdf_set_ref
                                      )

# # # ------------------------------------------------------------------------------------------------------------------ #
# # # --------------------------------------- Train GPR -------------------------------------------------------- #
# # # ------------------------------------------------------------------------------------------------------------------ #



    
# # Part 1. Initialization of information  ------------------------------------------------------------------------------
BME = np.zeros((iteration_limit+1, 1))
RE = np.zeros((iteration_limit+1, 1))
IE = np.zeros((iteration_limit+1, 1))
crit_val_BME = np.zeros((iteration_limit+1, 1))
crit_val_RE = np.zeros((iteration_limit+1, 1))
crit_val_IE = np.zeros((iteration_limit+1, 1))

h_BME = np.zeros((iteration_limit+1, 1))
h_RE = np.zeros((iteration_limit+1, 1))
h_IE = np.zeros((iteration_limit+1, 1))

c_BME = np.zeros((iteration_limit+1, 1))
c_RE = np.zeros((iteration_limit+1, 1))
c_IE = np.zeros((iteration_limit+1, 1))

graph_list = []
graph_name = []
crit_val = np.zeros((iteration_limit, 1))


crit_val_s_std = np.zeros((n_obs, iteration_limit+1))
crit_val_s_pre = np.zeros((n_obs,iteration_limit+1))       

val_s_std = np.zeros((n_obs, iteration_limit+1))
val_s_pre = np.zeros((n_obs,iteration_limit+1)) 


# # Loop for Bayesian update
for Ntp in range(0, iteration_limit+1):
    # surrogate_prediction = np.zeros((n_obs,MCMC))
    # surrogate_std = np.zeros((n_obs,MCMC))
    surrogate_prediction = np.zeros((n_obs,mc_size))
    surrogate_std = np.zeros((n_obs,mc_size))
    
    
    surrogate_prediction_MC_old = np.zeros((n_obs,1))
    surrogate_prediction_MC_new = np.zeros((n_obs,1)) 
    surrogate_std_MC_old = np.zeros((n_obs,1))
    surrogate_std_MC_new = np.zeros((n_obs,1))
    

    
    # surrogate_prediction_MC_old = np.zeros((n_obs,1))
    # surrogate_prediction_MC_new = np.zeros((n_obs,1))

    surrogate_prediction_MC = np.zeros((n_obs,MCMC))
    surrogate_std_MC = np.zeros((n_obs,MCMC))


    gp_list = []
    

    # Loop for each observation, which gets its own surrogate model
    


    
    for i, model in enumerate(eva.T):
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
        surrogate_prediction[i, :], surrogate_std[i, :] = gp.predict(prior_ref, return_std=True)
        # Save trained GP to use later
        gp_list.append(gp)
        # =============================================================================
        #         print(f"    Looping through loc {i}/{n_obs} --> hp: {gp.kernel_.theta}")
        # Evaluate all prior parameter sets in GP

        

        # with open((model_output_path+ '/output/gp.txt' ), "wb") as file:
        #     pickle.dump(gp_list, file)


        surrogate_prediction_h = np.array(surrogate_prediction.T)[0:mc_size,0:10]
        surrogate_prediction_c = np.array(surrogate_prediction.T)[0:mc_size,10:20]

        # hk_old = stats.multivariate_normal(mean = math.log(1e-5),cov=1).rvs(6)
       

        
        
        total_error_h = (h_measurement_error ** 2) * np.ones(h_obs.shape[0])
        h_BME[Ntp], h_RE[Ntp], h_IE[Ntp] = compute_bme(surrogate_prediction_h, 
                                            h_obs, 
                                            total_error_h,
                                            prior_pdf=prior_pdf_set_ref, 
                                            prior_samples=prior_pdf_set_ref)
        total_error_c = (conc_measurement_error ** 2) * np.ones(h_obs.shape[0])
        c_BME[Ntp], c_RE[Ntp], c_IE[Ntp] = compute_bme(surrogate_prediction_c, 
                                                  conc_obs, 
                                                  total_error_c,
                                                  prior_pdf=prior_pdf_set_ref, 
                                                  prior_samples=prior_pdf_set_ref)
        
        total_error = (measurement_error ** 2) * np.ones(obs.shape[0])
        BME[Ntp], RE[Ntp], IE[Ntp] = compute_bme(surrogate_prediction.T, 
                                                  obs, 
                                                  total_error,
                                                  prior_pdf=prior_pdf_set_ref, 
                                                  prior_samples=prior_pdf_set_ref)
        
        # hk_old = obs_hk[-1].reshape(1, 6)[0]
        
        # old_pdf_ind = stats.norm.pdf(hk_old, loc = math.log(1e-5), scale = 1)
        # old_pdf_set = np.sum(np.prod(old_pdf_ind))                                   
            
        # surrogate_prediction_MC_old= gp.predict(hk_old.reshape(1, -1))           
    
    val_s_std[:,Ntp] = surrogate_std[:,0]
    val_s_pre[:,Ntp] = surrogate_prediction[:,0]
    # hk_old = collocation_points[-1].reshape(1, 6)[0]
    
    hk_old = prior_ref[tp_i+Ntp] 
    # # surrogate_prediction_MC_old= gp.predict(hk_old)
    # surrogate_prediction_MC_old = [gp.predict(hk_old) for _ in range(20)]
    samples = np.zeros((MCMC,6))
    for MC in range(MCMC):

        

        adj_size = (MC+1) / 1000

        hk_new = stats.multivariate_normal(mean = hk_old.reshape(1, 6)[0], 
                                             cov=0.00009).rvs(1) 
                                            # 0.0008 0.00075 0.00085 0.00009 0.00007 0.000009



        
        for i, gp_model in enumerate(gp_list):
            surrogate_prediction_MC_new[i, :],surrogate_std_MC_new[i,:] = gp_model.predict(hk_new.reshape(1, -1), return_std=True)
            surrogate_prediction_MC_old[i, :],surrogate_std_MC_old[i,:]= gp_model.predict(hk_old.reshape(1, -1), return_std=True)


        
        new_pdf_ind = stats.norm.logpdf(hk_new, loc = math.log(1e-5), scale =  1)
        new_pdf_set = np.sum(new_pdf_ind)
        old_pdf_ind = stats.norm.logpdf(hk_old, loc = math.log(1e-5), scale =  1)
        old_pdf_set = np.sum(old_pdf_ind)

        
        llk_old = np.sum(stats.multivariate_normal.logpdf(surrogate_prediction_MC_old,  
                                                 cov=np.diag(total_error),
                                                 mean=obs[0, :]))
        llk_new = np.sum(stats.multivariate_normal.logpdf(surrogate_prediction_MC_new, 
                                                 cov= np.diag(total_error), 
                                                 mean= obs[0, :]))
        
        
        
        
        # prop_ratio = np.log(stats.multivariate_normal.pdf(hk_old, mean=np.mean(hk_new), cov=10)) - \
        #                 np.log(stats.multivariate_normal.pdf(hk_new,mean=np.mean(hk_old), cov=10))  
        # ll_diff = np.log(llk_new) - np.log(llk_old)
        # prior_diff = np.log(new_pdf_set) - np.log(old_pdf_set)
        α = np.exp(
            np.log(llk_new) + new_pdf_set - np.log(llk_old) - old_pdf_set
            ) #+ prop_ratio)
        
        if np.random.random() < min(1,α):
           samples[MC]  = hk_new
           hk_old = hk_new
           surrogate_prediction_MC[:,MC]=surrogate_prediction_MC_new[:, 0]
           surrogate_std_MC[:,MC]=surrogate_std_MC_new[:, 0]


        else:
            samples[MC]  = hk_old
            hk_old = hk_old
            surrogate_prediction_MC[:,MC]=surrogate_prediction_MC_old[:, 0]
            surrogate_std_MC[:,MC]=surrogate_std_MC_old[:, 0]

 

    prior = samples
    prior = np.array(prior)
    
    
    
    # # Get prior probability for each independent parameter value in each set
    # prior_pdf_ind = stats.norm.pdf(prior, loc = np.mean(prior), scale =np.std(prior))
    # # Get prior probability for each set
    # prior_pdf_set = np.prod(prior_pdf_ind, axis=1)
    # surrogate_prediction_MC, surrogate_std_MC = zip(*[gp.predict(prior, return_std=True) for _ in range(20)])
    # surrogate_prediction_MC = np.array(surrogate_prediction_MC)
    # surrogate_std_MC = np.array(surrogate_std_MC)
    
    # with open((model_output_path+ '/output/gp_list.pkl' ), 'rb') as file:
    #     loaded_gp_list = pickle.load(file)
    # for i, gp_model in enumerate(gp_list):
    #     surrogate_prediction_MC[i,:], surrogate_std_MC[i,:] = gp_model.predict(prior, return_std=True)

        
    
    # # Bayesian active learning (in output space) --------------------------------------------------------------
    if Ntp < iteration_limit:
       
        
        

        # Calculate the PDF for each data point (column) in surrogate_prediction
        aux1 = np.where((prior[:d_size_AL + Ntp, :] == collocation_points[:, None]).all(-1))[1]
        aux2 = np.invert(np.in1d(np.arange(prior[:d_size_AL + Ntp, :].shape[0]), aux1))
        al_unique_index = np.arange(prior[:d_size_AL + Ntp, :].shape[0])[aux2]
        # al_unique_index = 1000

        
        
        al_BME = np.zeros((len(al_unique_index), 1))
        al_RE = np.zeros((len(al_unique_index), 1))
        al_IE = np.zeros((len(al_unique_index), 1))
        
        # al_BME = np.zeros((1000, 1))
        # al_RE = np.zeros((1000, 1))
        # al_IE = np.zeros((1000, 1))
        
        # Index of the elements of the prior distribution that have not been used as collocation points

                
        
        
        for iAL in range(0, len(al_unique_index)):
            
        # for iAL in range(0, 1000):

            
            # Exploration of output subspace associated with a defined prior combination.
            al_exploration = np.random.normal(size=(mc_size_AL, n_obs)) * surrogate_std_MC[:, iAL] + \
                              surrogate_prediction_MC[:, iAL]
            
            # a11 = surrogate_prediction_MC[:, iAL]

        
                           
            proir_bal_pdf=stats.multivariate_normal.pdf(al_exploration, mean=surrogate_prediction[:, al_unique_index[iAL]],cov = surrogate_std[:, al_unique_index[iAL]])                 
           
            # BAL scores computation
            total_error = (measurement_error ** 2) * np.ones(obs.shape[0])
            al_BME[iAL], al_RE[iAL], al_IE[iAL] = compute_bme(model_predictions=al_exploration,
                                                              observations=obs,
                                                              var=total_error,
                                                              prior_pdf=proir_bal_pdf, 
                                                              prior_samples=al_exploration)

        
        

        

        
        # Part 7. Selection criteria for next collocation point ------------------------------------------------------
        al_value, al_value_index = bal_selection_criteria(al_strategy, al_BME, al_RE, al_IE)
        crit_val[Ntp, 0] = al_value
        crit_val_BME[Ntp]= al_BME[al_value_index]
        crit_val_RE[Ntp]= al_RE[al_value_index]
        crit_val_IE[Ntp]= al_IE[al_value_index]
        


        
        crit_val_s_std[:,Ntp] = surrogate_std_MC[:,al_unique_index[al_value_index]]
        crit_val_s_pre[:,Ntp] = surrogate_prediction_MC[:,al_unique_index[al_value_index]]
        # Part 8. Selection of new collocation point


        # collocation_points = np.vstack((collocation_points, prior[al_value_index, :]))
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
        h_eva1 = pd.read_csv((os.getcwd() + f'/model/{sim_name}/{gwfname}.obs.head.csv'), dtype=np.float64, header=0, usecols=range(1, 11), skiprows=1).iloc[-1:]
        h_eva1 = np.array(h_eva1).reshape(1,10)
        # h_eva1  = np.array(h_eva1)[:,0:6]
        h_eva1 = add_noise(h_eva1, h_error,n_sizes = 10, seed=0, ret_error=True)[0]


        
        h_eva = np.vstack((h_eva,h_eva1))
        
        conc_eva1 = pd.read_csv((os.getcwd() + f'/model/{sim_name}/{gwtname}.obs.conc.csv'),  dtype=np.float64, header=0, usecols=range(1, 11), skiprows=1).iloc[-1:]
        conc_eva1 = np.array(conc_eva1).reshape(1,10)
        # h_eva1  = np.array(h_eva1)[:,0:6]
        conc_eva1 = add_noise(conc_eva1, c_error,n_sizes = 10, seed=0, ret_error=True)[0]

        
        conc_eva = np.vstack((conc_eva,conc_eva1))
        eva = np.hstack((h_eva, conc_eva))
        
        
        
    # Progress report
    print("Bayesian iteration: " + str(Ntp + 1) + "/" + str(iteration_limit))

df = pd.DataFrame(collocation_points)        
df.to_csv(os.path.join(model_output_path+ r'/output','collocation_points.csv'))

df = pd.DataFrame(eva)        
df.to_csv(os.path.join(model_output_path+ r'/output','comb_TP.csv'))          
df = pd.DataFrame(surrogate_prediction)        
df.to_csv(os.path.join(model_output_path+ r'/output','surrogate_prediction.csv'))

df = pd.DataFrame(h_BME)        
df.to_csv(os.path.join(model_output_path+ r'/output','h_BME.csv'))
df = pd.DataFrame(c_BME)        
df.to_csv(os.path.join(model_output_path+ r'/output','c_BME.csv'))
df = pd.DataFrame(h_RE)        
df.to_csv(os.path.join(model_output_path+ r'/output','h_RE.csv'))
df = pd.DataFrame(c_RE)        
df.to_csv(os.path.join(model_output_path+ r'/output','c_RE.csv'))
df = pd.DataFrame(h_IE)        
df.to_csv(os.path.join(model_output_path+ r'/output','h_IE.csv'))
df = pd.DataFrame(c_IE)        
df.to_csv(os.path.join(model_output_path+ r'/output','c_IE.csv'))

df = pd.DataFrame(crit_val_BME)        
df.to_csv(os.path.join(model_output_path+ r'/output','crit_val_BME.csv'))
df = pd.DataFrame(crit_val_RE)        
df.to_csv(os.path.join(model_output_path+ r'/output','crit_val_RE.csv'))
df = pd.DataFrame(crit_val_IE)        
df.to_csv(os.path.join(model_output_path+ r'/output','crit_val_IE.csv'))

df = pd.DataFrame(crit_val_s_std)        
df.to_csv(os.path.join(model_output_path+ r'/output','crit_val_s_std.csv'))
df = pd.DataFrame(crit_val_s_pre)        
df.to_csv(os.path.join(model_output_path+ r'/output','crit_val_s_pre.csv'))

df = pd.DataFrame(val_s_std)        
df.to_csv(os.path.join(model_output_path+ r'/output','val_s_std.csv'))
df = pd.DataFrame(val_s_pre)        
df.to_csv(os.path.join(model_output_path+ r'/output','val_s_pre.csv'))

with open((model_output_path+ '/output/gp_list.pkl' ), 'wb') as file:
    pickle.dump(gp_list, file)
    


c_ref_bme, c_ref_re, c_ref_ie = compute_bme(
                                            model_predictions=conc_ref_output, 
                                            observations=conc_obs, 
                                            var=conc_measurement_error**2,
                                            prior_pdf=prior_pdf_set_ref,
                                            prior_samples=prior_pdf_set_ref)

h_ref_bme, h_ref_re, h_ref_ie = compute_bme(model_predictions=h_ref_output, 
                                            observations=h_obs, 
                                            var=h_measurement_error**2,
                                            prior_pdf=prior_pdf_set_ref, 
                                            prior_samples=prior_pdf_set_ref)


non_h = h_BME / h_ref_bme
non_h2 = h_IE / h_ref_ie
non_h3 = h_RE / h_ref_re

non_c = c_BME / c_ref_bme
non_c2 = c_IE / c_ref_ie
non_c3 = c_RE / c_ref_re





import matplotlib.pyplot as plt


fig, (ax1,ax2)= plt.subplots(2,1)
data1 = prior[0:-1,2]
# Create histograms for both datasets
ax1.hist(data1, bins=20, alpha=0.8, label='MCMC-Sampling', color='#0A65B6')
ax1.legend()
ax1.set_title('log(HK) Distribution of Zone 3')
ax1.set_xlabel('log - Hydraulic Conductivity')
ax2.plot(data1)
ax2.set_ylabel('log - Hydraulic Conductivity')
filename = "log(HK) Distribution of Zone 3.png" 
plt.savefig((model_output_path+ f'/output/{filename}'))
plt.show()

fig, (ax1,ax2)= plt.subplots(2,1)
data1 = prior[0:-1,5]
# Create histograms for both datasets
ax1.hist(data1, bins=20, alpha=0.8, label='MCMC-Sampling', color='#0A65B6')
ax1.legend()
# Customize the plot
ax1.set_title('Distribution of Zone 6')
ax1.set_xlabel('log - Hydraulic Conductivity')
# Create a line plot for the data
ax2.plot(data1)
ax2.set_ylabel('log - Hydraulic Conductivity')
filename = "HK Distribution of Zone 6.png" 
plt.savefig((model_output_path+ f'/output/{filename}'))
plt.show()

# fig, ((ax1,ax2),(ax3,ax4))= plt.subplots(2,2)
tname = f'Normalized BME for {tp_i} initial TP and {iteration_limit} BAL iterations with MCMC Sampling'
fig, (ax1,ax2)= plt.subplots(2,1)
fig.tight_layout(pad=1.0)
ax1.set_title('Hydraulic-head BAL')
ax1.plot(non_h, color='#61B2D7')
ax2.set_title('Concentration BAL')
ax2.plot(non_c, color='#D76177')
fig.suptitle(tname)
fig.subplots_adjust(top=0.9)
filename = "figure_Normalized_BME.png" 
plt.savefig((model_output_path+ f'/output/{filename}'))
plt.show() 



tname = f'Normalized IE for {tp_i} initial TP and {iteration_limit} BAL iterations with MCMC Sampling'
fig, (ax1,ax2)= plt.subplots(2,1)
fig.tight_layout(pad=1.0)
ax1.set_title('Hydraulic-head')
ax1.plot(non_h2, color='#61B2D7')
ax2.set_title('Concentration')
ax2.plot(non_c2, color='#D76177')
fig.suptitle(tname)
fig.subplots_adjust(top=0.9) 
filename = "figure_Normalized_IE.png" 
plt.savefig((model_output_path+ f'/output/{filename}'))
plt.show() 


tname = f'Normalized RE for {tp_i} initial TP and {iteration_limit} BAL iterations with MCMC Sampling'
fig, (ax1,ax2)= plt.subplots(2,1)
fig.tight_layout(pad=1.0)
ax1.set_title('Hydraulic-head')
ax1.plot(non_h3, color='#61B2D7')
ax2.set_title('Concentration')
ax2.plot(non_c3, color='#D76177')
fig.suptitle(tname)
fig.subplots_adjust(top=0.9) 
filename = "figure_Normalized_RE.png" 
plt.savefig((model_output_path+ f'/output/{filename}'))
plt.show()  
# Plot results:
# Get confidence intervals
lower_ci = surrogate_prediction - 2*surrogate_std
upper_ci = surrogate_prediction + 2*surrogate_std

if n_zones  == 1:
    plot_1d_gpe_final(prior, surrogate_prediction,  lower_ci, upper_ci, collocation_points,
                        eva, tp_i, iteration_limit,  obs)

plot_gpe_scores(BME, RE, tp_i, ref_bme= ref_bme, ref_re = ref_re)
plot_bal_criteria(crit_val, al_strategy)

Tf = time.time()
Total_running = ((Tf - T)/60/60)
running_time_str = 'Total running: {:.3f} h'.format(Total_running)
print('Total running:%s h' % ((Tf - T)/60/60))

with open((model_output_path+ '/output/runningtime.txt' ), 'w') as file:
    file.write(running_time_str)
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


# c  =  zones_vec
# fig, (ax1)= plt.subplots()
# cmap = plt.get_cmap('#09ACF3', 6)
# mod_zones01 = c#[0 : nrow, 0 : ncol]
# im3 = ax1.imshow(mod_zones01, cmap=cmap
#                  #['#F309AC', '#F30937', '#09ACF3', '#F3C509', '#ACF309','#9703FF']) 
#                  )
# ax1.set_title("Zone Distribution")
# bounds = [1, 2, 3, 4, 5, 6]
# cbar = fig.colorbar(im3, ax=ax1,  ticks=bounds, spacing='proportional')


# ax1.scatter(point_loc[:, 1], point_loc[:, 0], marker='o',c='#FFFFFF')        
# plt.show()  