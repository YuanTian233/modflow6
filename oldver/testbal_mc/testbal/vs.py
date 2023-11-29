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
import scipy.io
from bayesian_inference import calculate_likelihood, calculate_likelihood_manual, compute_bme, bal_selection_criteria

import matplotlib.pyplot as plt

import arviz as az

az.style.use("arviz-doc")





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

cre ='#61D786'
cie ='#61B2D7'
cbme ='#D76177'


color1='#61B2D7'
color2='#D76177'



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
iteration_limit = 270



n_zones = 6 
n_reali =  1000


gm_mean = math.log(1e-5)  # ln(K) average --> constant

    
zones_vec50x50 = pd.read_csv((model_output_path + '/input/zone_distribution.csv'), dtype=np.float64)
zones_vec = np.array(zones_vec50x50, dtype = float)





point_loc = np.array(
    [(25, 25),
    (24, 31),
    (45, 18),
    (28, 7),
    (32, 27),
    (8, 36),
    (7, 10),
    (18, 20),
    (28, 32),
    (39, 41)]
    ) # this mark the location of obs points on plot

fig, ax1 = plt.subplots()

# Plot zones
cmap = plt.get_cmap('GnBu', 6)
im3 = ax1.imshow(zones_vec, cmap=cmap)


bounds = [1, 2, 3, 4, 5, 6]
cbar = fig.colorbar(im3, ax=ax1, 
                    ticks=bounds, 
                    spacing='uniform'
                    )

# Customize color bar ticks and labels
cbar.set_ticks([1, 2, 3, 4, 5, 6])
cbar.set_ticklabels(['Zone 1', 'Zone 2', 'Zone 3', 'Zone 4', 'Zone 5', 'Zone 6'])



# Plot observation points
ax1.scatter(point_loc[:, 1], point_loc[:, 0], marker='o', c='#112F2C', edgecolor='w', s=80)
ax1.scatter(25, 25, marker='o', c='red', edgecolor='w', s=80)

# Save the plot
filename = "figure_zones.png"
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")

# Show the plot
plt.show()  


obs = pd.read_csv((model_output_path + '/input/obs_hk.csv'),header = 0, index_col = 0)
obs= np.array(obs)


mat_file_path = (model_output_path + '/ref_data/Y_true.mat')

# Load the .mat file
mat_contents = scipy.io.loadmat(mat_file_path)
y_true = mat_contents['Y_true']

# Assuming point_loc is defined earlier in your code
fig, ax1 = plt.subplots()
cmap = plt.get_cmap('GnBu')
im3 = ax1.imshow(y_true, cmap=cmap)
ax1.scatter(point_loc[:, 1], point_loc[:, 0], marker='o', c='#112F2C', edgecolor='w', s=80)
ax1.scatter(25, 25, marker='o', c='red', edgecolor='w', s=80)
cbar = plt.colorbar(im3, label='log(HK)')
filename = "figure_y_true.png"
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")

# Show the plot
plt.show()


# Create a 50x50 grid
grid_size = 50
grid = np.zeros((grid_size, grid_size))  # Initialize the grid with zeros

# Set the first column (column index 0) to red (value 1.0)
grid[:, 0] = 1.0
grid[:, -1] = 1.0
grid[20:31, 0] = 0.5
# # Create a custom colormap with red for value 1.0 and white for value 0.0
from matplotlib.colors import ListedColormap
colors = ['white', cbme,cie]
cmap = ListedColormap(colors)
# Create a figure and axis for the map
fig, ax = plt.subplots(figsize=(8,6.5))

# Display the grid as a map with the custom colormap
im = ax.imshow(grid, cmap=cmap, interpolation='none', aspect='auto', extent=[0, grid_size, 0, grid_size])



# Set axis labels and title
ax.set_xlabel('No Flow', weight='bold')
ax.set_ylabel('H = 1 m', weight='bold')
ax.set_title('No Flow',  weight='bold')
ax_right = ax.twinx()
ax.scatter(point_loc[:, 1], point_loc[:, 0], marker='o', c='#112F2C', edgecolor='w', s=80)
ax_right.set_yticks(np.arange(0, grid_size+1, 10))
ax_right.set_ylabel('H = 0 m',  weight='bold')
# Show the map
filename = "figure_50by50.png"
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()


fig, ax1 = plt.subplots()
cmap = plt.get_cmap()
from gwm import hkfields
k = hkfields(zones_vec = zones_vec,parameter_sets = obs,n_zones = n_zones)
im3 = ax1.imshow(k, cmap=cmap)
ax1.scatter(point_loc[:, 1], point_loc[:, 0], marker='o', c='w', edgecolor='b', s=80)
ax1.scatter(25, 25, marker='o', c='red', edgecolor='w', s=80)
cbar = plt.colorbar(im3, label='log(HK)')
filename = "figure_obs_hk.png"
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")

# Show the plot
plt.show()






ref_h = np.full(270,h_ref_output [0,0])
ref_c = np.full(270,conc_ref_output [0,0])

prior_mc_bme_mean = np.array(pd.read_csv('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/BME/crit_val_s_pre.csv', dtype=np.float64, header=0, index_col=0))
prior_mc_bme_std = np.array(pd.read_csv('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/BME/crit_val_s_std.csv', dtype=np.float64, header=0, index_col=0))
prior_mc_re_mean = np.array(pd.read_csv('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/RE/crit_val_s_pre.csv', dtype=np.float64, header=0, index_col=0))
prior_mc_re_std = np.array(pd.read_csv('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/RE/crit_val_s_std.csv', dtype=np.float64, header=0, index_col=0))
prior_mc_ie_mean = np.array(pd.read_csv('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/IE/crit_val_s_pre.csv', dtype=np.float64, header=0, index_col=0))
prior_mc_ie_std = np.array(pd.read_csv('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/IE/crit_val_s_std.csv', dtype=np.float64, header=0, index_col=0))


prior_mh_bme_mean = np.array(pd.read_csv('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/BME/crit_val_s_pre.csv', dtype=np.float64, header=0, index_col=0))
prior_mh_bme_std = np.array(pd.read_csv('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/BME/crit_val_s_std.csv', dtype=np.float64, header=0, index_col=0))
prior_mh_re_mean = np.array(pd.read_csv('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/RE/crit_val_s_pre.csv', dtype=np.float64, header=0, index_col=0))
prior_mh_re_std = np.array(pd.read_csv('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/RE/crit_val_s_std.csv', dtype=np.float64, header=0, index_col=0))
prior_mh_ie_mean = np.array(pd.read_csv('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/IE/crit_val_s_pre.csv', dtype=np.float64, header=0, index_col=0))
prior_mh_ie_std = np.array(pd.read_csv('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/IE/crit_val_s_std.csv', dtype=np.float64, header=0, index_col=0))


post_mh_bme_mean = np.array(pd.read_csv('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/post/BME/crit_val_s_pre.csv', dtype=np.float64, header=0, index_col=0))
post_mh_bme_std = np.array(pd.read_csv('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/post/BME/crit_val_s_std.csv', dtype=np.float64, header=0, index_col=0))
post_mh_re_mean = np.array(pd.read_csv('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/post/RE/crit_val_s_pre.csv', dtype=np.float64, header=0, index_col=0))
post_mh_re_std = np.array(pd.read_csv('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/post/RE/crit_val_s_std.csv', dtype=np.float64, header=0, index_col=0))
post_mh_ie_mean = np.array(pd.read_csv('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/post/IE/crit_val_s_pre.csv', dtype=np.float64, header=0, index_col=0))
post_mh_ie_std = np.array(pd.read_csv('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/post/IE/crit_val_s_std.csv', dtype=np.float64, header=0, index_col=0))


post_mc_bme_mean = np.array(pd.read_csv('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/post/BME/crit_val_s_pre.csv', dtype=np.float64, header=0, index_col=0))
post_mc_bme_std = np.array(pd.read_csv('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/post/BME/crit_val_s_std.csv', dtype=np.float64, header=0, index_col=0))
post_mc_re_mean = np.array(pd.read_csv('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/post/RE/crit_val_s_pre.csv', dtype=np.float64, header=0, index_col=0))
post_mc_re_std = np.array(pd.read_csv('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/post/RE/crit_val_s_std.csv', dtype=np.float64, header=0, index_col=0))
post_mc_ie_mean = np.array(pd.read_csv('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/post/IE/crit_val_s_pre.csv', dtype=np.float64, header=0, index_col=0))
post_mc_ie_std = np.array(pd.read_csv('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/post/IE/crit_val_s_std.csv', dtype=np.float64, header=0, index_col=0))



figsize = (10, 6.18)
tname = f'figure_prior_mc_re_h_c'

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)
# Fill between lower and upper confidence intervals
mean = prior_mc_re_mean[0,0:270]
std = prior_mc_re_std[0,0:270]
lower_ci = mean - 2 * std
upper_ci = mean + 2 * std
ax1.plot(mean, label='E(h_{$MC^{RE^{BAL}}}$)',color=color1)
ax1.plot(ref_h, label='REF',color=color2)
ax1.fill_between(range(len(mean)), lower_ci, upper_ci, color=color1, alpha=0.3, label='95% CI')
ax1.legend()
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel('Hydraulic head (m)')

mean = prior_mc_re_mean[10,0:270]
std = prior_mc_re_std[10,0:270]
lower_ci = mean - 2 * std
upper_ci = mean + 2 * std
ax2.plot(mean, label='E(c_{$MC^{RE^{BAL}}}$)',color=color1)
ax2.plot(ref_c, label='REF',color=color2)
ax2.fill_between(range(len(mean)), lower_ci, upper_ci, color=color1, alpha=0.3, label='95% CI')
ax2.legend()
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel('Concentration (mg/L)')

# Save the plot to a file
filename = "figure_prior_mc_re_h_c.png"
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()

tname = f'figure_prior_mc_ie_h_c'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)
# Fill between lower and upper confidence intervals
mean = prior_mc_ie_mean[0,0:270]
std = prior_mc_ie_std[0,0:270]
lower_ci = mean - 2 * std
upper_ci = mean + 2 * std
ax1.plot(mean, label='E(h_{$MC^{IE^{BAL}}}$)',color=color1)
ax1.plot(ref_h, label='REF',color=color2)
ax1.fill_between(range(len(mean)), lower_ci, upper_ci, color=color1, alpha=0.3, label='95% CI')
ax1.legend()
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel('Hydraulic head (m)')

mean = prior_mc_ie_mean[10,0:270]
std = prior_mc_ie_std[10,0:270]
lower_ci = mean - 2 * std
upper_ci = mean + 2 * std
ax2.plot(mean, label='E(c_{$MC^{IE^{BAL}}}$)',color=color1)
ax2.plot(ref_c, label='REF',color=color2)
ax2.fill_between(range(len(mean)), lower_ci, upper_ci, color=color1, alpha=0.3, label='95% CI')
ax2.legend()
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel('Concentration (mg/L)')
# Save the plot to a file
filename = "figure_prior_mc_ie_h_c.png"
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()


tname = f'figure_prior_mc_bme_h_c'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)
# Fill between lower and upper confidence intervals
mean = prior_mc_bme_mean[0,0:270]
std = prior_mc_bme_std[0,0:270]
lower_ci = mean - 2 * std
upper_ci = mean + 2 * std
ax1.plot(mean, label='E(h_{$MC^{BME^{BAL}}}$)',color=color1)
ax1.plot(ref_h, label='REF',color=color2)
ax1.fill_between(range(len(mean)), lower_ci, upper_ci, color=color1, alpha=0.3, label='95% CI')
ax1.legend()
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel('Hydraulic head (m)')

mean = prior_mc_bme_mean[10,0:270]
std = prior_mc_bme_std[10,0:270]
lower_ci = mean - 2 * std
upper_ci = mean + 2 * std
ax2.plot(mean, label='E(c_{$MC^{BME^{BAL}}}$)',color=color1)
ax2.plot(ref_c, label='REF',color=color2)
ax2.fill_between(range(len(mean)), lower_ci, upper_ci, color=color1, alpha=0.3, label='95% CI')
ax2.legend()
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel('Concentration (mg/L)')
# Save the plot to a file
filename = "figure_prior_mc_bme_h_c.png"
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()




tname = f'figure_post_mc_re_h_c'

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)
# Fill between lower and upper confidence intervals
mean = post_mc_re_mean[0,0:270]
std = post_mc_re_std[0,0:270]
lower_ci = mean - 2 * std
upper_ci = mean + 2 * std
ax1.plot(mean, label='E(h_{$MC^{RE^{BAL}}}$)',color=color1)
ax1.plot(ref_h, label='REF',color=color2)
ax1.fill_between(range(len(mean)), lower_ci, upper_ci, color=color1, alpha=0.3, label='95% CI')
ax1.legend()
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel('Hydraulic head (m)')

mean = post_mc_re_mean[10,0:270]
std = post_mc_re_std[10,0:270]
lower_ci = mean - 2 * std
upper_ci = mean + 2 * std
ax2.plot(mean, label='E(c_{$MC^{RE^{BAL}}}$)',color=color1)
ax2.plot(ref_c, label='REF',color=color2)
ax2.fill_between(range(len(mean)), lower_ci, upper_ci, color=color1, alpha=0.3, label='95% CI')
ax2.legend()
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel('Concentration (mg/L)')

# Save the plot to a file
filename = "figure_post_mc_re_h_c.png"
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()

tname = f'figure_post_mc_ie_h_c'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)
# Fill between lower and upper confidence intervals
mean = post_mc_ie_mean[0,0:270]
std = post_mc_ie_std[0,0:270]
lower_ci = mean - 2 * std
upper_ci = mean + 2 * std
ax1.plot(mean, label='E(h_{$MC^{IE^{BAL}}}$)',color=color1)
ax1.plot(ref_h, label='REF',color=color2)
ax1.fill_between(range(len(mean)), lower_ci, upper_ci, color=color1, alpha=0.3, label='95% CI')
ax1.legend()
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel('Hydraulic head (m)')

mean = post_mc_ie_mean[10,0:270]
std = post_mc_ie_std[10,0:270]
lower_ci = mean - 2 * std
upper_ci = mean + 2 * std
ax2.plot(mean, label='E(c_{$MC^{IE^{BAL}}}$)',color=color1)
ax2.plot(ref_c, label='REF',color=color2)
ax2.fill_between(range(len(mean)), lower_ci, upper_ci, color=color1, alpha=0.3, label='95% CI')
ax2.legend()
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel('Concentration (mg/L)')
# Save the plot to a file
filename = "figure_post_mc_ie_h_c.png"
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()


tname = f'figure_post_mc_bme_h_c'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)
# Fill between lower and upper confidence intervals
mean = post_mc_bme_mean[0,0:270]
std = post_mc_bme_std[0,0:270]
lower_ci = mean - 2 * std
upper_ci = mean + 2 * std
ax1.plot(mean, label='E(h_{$MC^{BME^{BAL}}}$)',color=color1)
ax1.plot(ref_h, label='REF',color=color2)
ax1.fill_between(range(len(mean)), lower_ci, upper_ci, color=color1, alpha=0.3, label='95% CI')
ax1.legend()
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel('Hydraulic head (m)')

mean = post_mc_bme_mean[10,0:270]
std = post_mc_bme_std[10,0:270]
lower_ci = mean - 2 * std
upper_ci = mean + 2 * std
ax2.plot(mean, label='E(c_{$MC^{BME^{BAL}}}$)',color=color1)
ax2.plot(ref_c, label='REF',color=color2)
ax2.fill_between(range(len(mean)), lower_ci, upper_ci, color=color1, alpha=0.3, label='95% CI')
ax2.legend()
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel('Concentration (mg/L)')
# Save the plot to a file
filename = "figure_post_mc_bme_h_c.png"
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()


tname = f'figure_post_mh_re_h_c'

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)
# Fill between lower and upper confidence intervals
mean = post_mh_re_mean[0,0:270]
std = post_mh_re_std[0,0:270]
lower_ci = mean - 2 * std
upper_ci = mean + 2 * std
ax1.plot(mean, label='E(h_{$MCMC^{RE^{BAL}}}$)',color=color1)
ax1.plot(ref_h, label='REF',color=color2)
ax1.fill_between(range(len(mean)), lower_ci, upper_ci, color=color1, alpha=0.3, label='95% CI')
ax1.legend()
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel('Hydraulic head (m)')

mean = post_mh_re_mean[10,0:270]
std = post_mh_re_std[10,0:270]
lower_ci = mean - 2 * std
upper_ci = mean + 2 * std
ax2.plot(mean, label='E(c_{$MCMC^{RE^{BAL}}}$)',color=color1)
ax2.plot(ref_c, label='REF',color=color2)
ax2.fill_between(range(len(mean)), lower_ci, upper_ci, color=color1, alpha=0.3, label='95% CI')
ax2.legend()
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel('Concentration (mg/L)')

# Save the plot to a file
filename = "figure_post_mh_re_h_c.png"
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()

tname = f'figure_post_mh_ie_h_c'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)
# Fill between lower and upper confidence intervals
mean = post_mh_ie_mean[0,0:270]
std = post_mh_ie_std[0,0:270]
lower_ci = mean - 2 * std
upper_ci = mean + 2 * std
ax1.plot(mean, label='E(h_{$MCMC^{IE^{BAL}}}$)',color=color1)
ax1.plot(ref_h, label='REF',color=color2)
ax1.fill_between(range(len(mean)), lower_ci, upper_ci, color=color1, alpha=0.3, label='95% CI')
ax1.legend()
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel('Hydraulic head (m)')

mean = post_mh_ie_mean[10,0:270]
std = post_mh_ie_std[10,0:270]
lower_ci = mean - 2 * std
upper_ci = mean + 2 * std
ax2.plot(mean, label='E(c_{$MCMC^{IE^{BAL}}}$)',color=color1)
ax2.plot(ref_c, label='REF',color=color2)
ax2.fill_between(range(len(mean)), lower_ci, upper_ci, color=color1, alpha=0.3, label='95% CI')
ax2.legend()
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel('Concentration (mg/L)')
# Save the plot to a file
filename = "figure_post_mh_ie_h_c.png"
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()


tname = f'figure_post_mh_bme_h_c'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)
# Fill between lower and upper confidence intervals
mean = post_mh_bme_mean[0,0:270]
std = post_mh_bme_std[0,0:270]
lower_ci = mean - 2 * std
upper_ci = mean + 2 * std
ax1.plot(mean, label='E(h_{$MCMC^{BME^{BAL}}}$)',color=color1)
ax1.plot(ref_h, label='REF',color=color2)
ax1.fill_between(range(len(mean)), lower_ci, upper_ci, color=color1, alpha=0.3, label='95% CI')
ax1.legend()
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel('Hydraulic head (m)')

mean = post_mh_bme_mean[10,0:270]
std = post_mh_bme_std[10,0:270]
lower_ci = mean - 2 * std
upper_ci = mean + 2 * std
ax2.plot(mean, label='E(c_{$MCMC^{BME^{BAL}}}$)',color=color1)
ax2.plot(ref_c, label='REF',color=color2)
ax2.fill_between(range(len(mean)), lower_ci, upper_ci, color=color1, alpha=0.3, label='95% CI')
ax2.legend()
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel('Concentration (mg/L)')
# Save the plot to a file
filename = "figure_post_mh_bme_h_c.png"
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()


tname = f'figure_prior_mh_re_h_c'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)
# Fill between lower and upper confidence intervals
mean = prior_mh_re_mean[0,0:270]
std = prior_mh_re_std[0,0:270]
lower_ci = mean - 2 * std
upper_ci = mean + 2 * std
ax1.plot(mean, label='E(h_{$MCMC^{RE^{BAL}}}$)',color=color1)
ax1.plot(ref_h, label='REF',color=color2)
ax1.fill_between(range(len(mean)), lower_ci, upper_ci, color=color1, alpha=0.3, label='95% CI')
ax1.legend()
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel('Hydraulic head (m)')

mean = prior_mh_re_mean[10,0:270]
std = prior_mh_re_std[10,0:270]
lower_ci = mean - 2 * std
upper_ci = mean + 2 * std
ax2.plot(mean, label='E(c_{$MCMC^{RE^{BAL}}}$)',color=color1)
ax2.plot(ref_c, label='REF',color=color2)
ax2.fill_between(range(len(mean)), lower_ci, upper_ci, color=color1, alpha=0.3, label='95% CI')
ax2.legend()
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel('Concentration (mg/L)')

# Save the plot to a file
filename = "figure_prior_mh_re_h_c.png"
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()

tname = f'figure_prior_mh_ie_h_c'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)
# Fill between lower and upper confidence intervals
mean = prior_mh_ie_mean[0,0:270]
std = prior_mh_ie_std[0,0:270]
lower_ci = mean - 2 * std
upper_ci = mean + 2 * std
ax1.plot(mean, label='E(h_{$MCMC^{IE^{BAL}}}$)',color=color1)
ax1.plot(ref_h, label='REF',color=color2)
ax1.fill_between(range(len(mean)), lower_ci, upper_ci, color=color1, alpha=0.3, label='95% CI')
ax1.legend()
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel('Hydraulic head (m)')

mean = prior_mh_ie_mean[10,0:270]
std = prior_mh_ie_std[10,0:270]
lower_ci = mean - 2 * std
upper_ci = mean + 2 * std
ax2.plot(mean, label='E(c_{$MCMC^{IE^{BAL}}}$)',color=color1)
ax2.plot(ref_c, label='REF',color=color2)
ax2.fill_between(range(len(mean)), lower_ci, upper_ci, color=color1, alpha=0.3, label='95% CI')
ax2.legend()
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel('Concentration (mg/L)')
# Save the plot to a file
filename = "figure_prior_mh_ie_h_c.png"
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()


tname = f'figure_prior_mh_bme_h_c'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)
# Fill between lower and upper confidence intervals
mean = prior_mh_bme_mean[0,0:270]
std = prior_mh_bme_std[0,0:270]
lower_ci = mean - 2 * std
upper_ci = mean + 2 * std
ax1.plot(mean, label='E(h_{$MCMC^{BME^{BAL}}}$)',color=color1)
ax1.plot(ref_h, label='REF',color=color2)
ax1.fill_between(range(len(mean)), lower_ci, upper_ci, color=color1, alpha=0.3, label='95% CI')
ax1.legend()
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel('Hydraulic head (m)')

mean = prior_mh_bme_mean[10,0:270]
std = prior_mh_bme_std[10,0:270]
lower_ci = mean - 2 * std
upper_ci = mean + 2 * std
ax2.plot(mean, label='E(c_{$MCMC^{BME^{BAL}}}$)',color=color1)
ax2.plot(ref_c, label='REF',color=color2)
ax2.fill_between(range(len(mean)), lower_ci, upper_ci, color=color1, alpha=0.3, label='95% CI')
ax2.legend()
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel('Concentration (mg/L)')
# Save the plot to a file
filename = "figure_prior_mh_bme_h_c.png"
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()












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














"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
-------------Prior---MC----------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

figsize = (10, 6.18)

tname = f'Normalized Scores with Prior-based MC sampling'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)

ax1.set_title('Hydraulic head', fontname='Arial', fontsize=14)
line1, = ax1.plot(n_h_mc_bme_bme, color=cbme, label=r'$BME^{BAL}$')  
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
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()






tname = f'Normalized Scores with Prior-based MC sampling'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)

ax1.set_title('Hydraulic head', fontname='Arial', fontsize=14)
line1, = ax1.plot(n_h_mc_bme_re, color=cbme, label=r'$BME^{BAL}$')  
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
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()


tname = f'Normalized Scores with Prior-based MC sampling'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)

ax1.set_title('Hydraulic head', fontname='Arial', fontsize=14)
line1, = ax1.plot(n_h_mc_bme_ie, color=cbme, label=r'$BME^{BAL}$')  
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
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()






"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
-------------Post---MC----------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


tname = f'Normalized Scores with Post-based MC sampling'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)

ax1.set_title('Hydraulic head', fontname='Arial', fontsize=14)
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
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()








tname = f'Normalized Scores with Post-based MC sampling'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)

ax1.set_title('Hydraulic head', fontname='Arial', fontsize=14)
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
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()




tname = f'Normalized Scores with Post-based MC sampling'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)

ax1.set_title('Hydraulic head', fontname='Arial', fontsize=14)
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
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()








"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
-------------Prior---MCMC----------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


tname = f'Normalized Scores with Prior-based MCMC sampling'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)

ax1.set_title('Hydraulic head', fontname='Arial', fontsize=14)
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
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()

tname = f'Normalized Scores with Prior-based MCMC sampling'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)

ax1.set_title('Hydraulic head', fontname='Arial', fontsize=14)
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
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()

tname = f'Normalized Scores with Prior-based MCMC sampling'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)

ax1.set_title('Hydraulic head', fontname='Arial', fontsize=14)
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
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()



"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
-------------Post---MCMC----------------------------------
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""


tname = f'Normalized Scores with Post-based MCMC sampling'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)

ax1.set_title('Hydraulic head', fontname='Arial', fontsize=14)
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
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()

tname = f'Normalized Scores with Post-based MCMC sampling'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)

ax1.set_title('Hydraulic head', fontname='Arial', fontsize=14)
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
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()

tname = f'Normalized Scores with Post-based MCMC sampling'
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
fig.tight_layout(pad=1.0)
fig.subplots_adjust(hspace=0.4)

ax1.set_title('Hydraulic head', fontname='Arial', fontsize=14)
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
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()




# tname = f'Normalized Scores with Posterior-based MC sampling'
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 4.944))
# fig.tight_layout(pad=1.0)
# fig.subplots_adjust(hspace=0.25)
# ax1.set_title('Hydraulic head', fontname='Arial', fontsize=14)
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
# ax1.set_title('Hydraulic head', fontname='Arial', fontsize=14)
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

figsize = (12, 7.416)

tname = f'Normalized Scores with Prior-based sampling using BME Selection'
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=figsize )
fig.tight_layout(pad=3.0)
fig.subplots_adjust(hspace=0.5)

# Plot 1
# ax1.set_title('RE: Hydraulic head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax1.plot(n_h_mc_bme_re, color='#61B2D7', label=r'$MC^{BME^{BAL}}$', linewidth=1.5)  
line2, = ax1.plot(n_h_mh_bme_re, color='#D76177', label=r'$MCMC^{BME^{BAL}}$', linewidth=1.5)
ax1.legend(handles=[line1, line2], prop={'size': 10})
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel(r'$D_{KL}/D_{KL}^{REF}$')

# Plot 2
# ax2.set_title('IE: Hydraulic head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax2.plot(n_h_mc_bme_ie, color='#61B2D7', label=r'$MC^{BME^{BAL}}$', linewidth=1.5)  
line2, = ax2.plot(n_h_mh_bme_ie, color='#D76177', label=r'$MCMC^{BME^{BAL}}$', linewidth=1.5)
ax2.legend(handles=[line1, line2], prop={'size': 10})
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel(r'$H/H_{REF}$')

# Plot 3
# ax3.set_title('BME: Hydraulic head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax3.plot(n_h_mc_bme_bme, color='#61B2D7', label=r'$MC^{BME^{BAL}}$', linewidth=1.5)  
line2, = ax3.plot(n_h_mh_bme_bme, color='#D76177', label=r'$MCMC^{BME^{BAL}}$', linewidth=1.5)
ax3.legend(handles=[line1, line2], prop={'size': 10})
ax3.set_xlabel('Number of model runs')
ax3.set_ylabel(r'$BME/BME_{Ref}$')


# Plot 4
# ax4.set_title('RE: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax4.plot(n_c_mc_bme_re, color='#61B2D7', label=r'$MC^{BME^{BAL}}$', linewidth=1.5)  
line2, = ax4.plot(n_c_mh_bme_re, color='#D76177', label=r'$MCMC^{BME^{BAL}}$', linewidth=1.5)
ax4.legend(handles=[line1, line2], prop={'size': 10})
ax4.set_xlabel('Number of model runs')
ax4.set_ylabel(r'$D_{KL}/D_{KL}^{REF}$')

# Plot 5
# ax5.set_title('IE: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax5.plot(n_c_mc_bme_ie, color='#61B2D7', label=r'$MC^{BME^{BAL}}$', linewidth=1.5)  
line2, = ax5.plot(n_c_mh_bme_ie, color='#D76177', label=r'$MCMC^{BME^{BAL}}$', linewidth=1.5)
ax5.legend(handles=[line1, line2], prop={'size': 10})
ax5.set_xlabel('Number of model runs')
ax5.set_ylabel(r'$H/H_{REF}$')

# Plot 6
# ax6.set_title('BME: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax6.plot(n_c_mc_bme_bme, color='#61B2D7', label=r'$MC^{BME^{BAL}}$', linewidth=1.5)  
line2, = ax6.plot(n_c_mh_bme_bme, color='#D76177', label=r'$MCMC^{BME^{BAL}}$', linewidth=1.5)
ax6.legend(handles=[line1, line2], prop={'size': 10})
ax6.set_xlabel('Number of model runs')
ax6.set_ylabel(r'$BME/BME_{Ref}$')

tname1 = 'Hydraulic head'
tname2 = 'Concentration'

# Subtitle for the first three subplots
fig.text(0.5, 0.94, tname1,  ha='center',fontsize=12, fontweight='bold', fontname='Arial')

# Subtitle for the last three subplots
fig.text(0.5, 0.48, tname2, ha='center', fontsize=12, fontweight='bold', fontname='Arial')



filename = "figure_Normalized_bme_comp.png"
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()


tname = f'Normalized Scores with Prior-based sampling using RE Selection'
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=figsize )
fig.tight_layout(pad=3.0)
fig.subplots_adjust(hspace=0.5)

# Plot 1
# ax1.set_title('RE: Hydraulic head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax1.plot(n_h_mc_re_re, color='#61B2D7', label=r'$MC^{D_{KL}^{BAL}}$', linewidth=1.5)  
line2, = ax1.plot(n_h_mh_re_re, color='#D76177', label=r'$MCMC^{D_{KL}^{BAL}}$', linewidth=1.5)
ax1.legend(handles=[line1, line2], prop={'size': 10})
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel(r'$D_{KL}/D_{KL}^{REF}$')


# Plot 2
# ax2.set_title('IE: Hydraulic head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax2.plot(n_h_mc_re_ie, color='#61B2D7', label=r'$MC^{D_{KL}^{BAL}}$', linewidth=1.5)  
line2, = ax2.plot(n_h_mh_re_ie, color='#D76177', label=r'$MCMC^{D_{KL}^{BAL}}$', linewidth=1.5)
ax2.legend(handles=[line1, line2], prop={'size': 10})
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel(r'$H/H_{REF}$')


# Plot 3
# ax3.set_title('BME: Hydraulic head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax3.plot(n_h_mc_re_bme, color='#61B2D7', label=r'$MC^{D_{KL}^{BAL}}$', linewidth=1.5)  
line2, = ax3.plot(n_h_mh_re_bme, color='#D76177', label=r'$MCMC^{D_{KL}^{BAL}}$', linewidth=1.5)
ax3.legend(handles=[line1, line2], prop={'size': 10})
ax3.set_xlabel('Number of model runs')
ax3.set_ylabel(r'$BME/BME_{Ref}$')

# Plot 4
# ax4.set_title('RE: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax4.plot(n_c_mc_re_re, color='#61B2D7', label=r'$MC^{D_{KL}^{BAL}}$', linewidth=1.5)  
line2, = ax4.plot(n_c_mh_re_re, color='#D76177', label=r'$MCMC^{D_{KL}^{BAL}}$', linewidth=1.5)
ax4.legend(handles=[line1, line2], prop={'size': 10})
ax4.set_xlabel('Number of model runs')
ax4.set_ylabel(r'$D_{KL}/D_{KL}^{REF}$')

# Plot 5
# ax5.set_title('IE: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax5.plot(n_c_mc_re_ie, color='#61B2D7', label=r'$MC^{D_{KL}^{BAL}}$', linewidth=1.5)  
line2, = ax5.plot(n_c_mh_re_ie, color='#D76177', label=r'$MCMC^{D_{KL}^{BAL}}$', linewidth=1.5)
ax5.legend(handles=[line1, line2], prop={'size': 10})
ax5.set_xlabel('Number of model runs')
ax5.set_ylabel(r'$H/H_{REF}$')

# Plot 6
# ax6.set_title('BME: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax6.plot(n_c_mc_re_bme, color='#61B2D7', label=r'$MC^{D_{KL}^{BAL}}$', linewidth=1.5)  
line2, = ax6.plot(n_c_mh_re_bme, color='#D76177', label=r'$MCMC^{D_{KL}^{BAL}}$', linewidth=1.5)
ax6.legend(handles=[line1, line2], prop={'size': 10})
ax6.set_xlabel('Number of model runs')
ax6.set_ylabel(r'$BME/BME_{Ref}$')


tname1 = 'Hydraulic head'
tname2 = 'Concentration'

# Subtitle for the first three subplots
fig.text(0.5, 0.94, tname1,  ha='center',fontsize=12, fontweight='bold', fontname='Arial')

# Subtitle for the last three subplots
fig.text(0.5, 0.48, tname2, ha='center', fontsize=12, fontweight='bold', fontname='Arial')
# fig.suptitle(tname, fontname='Arial', fontsize=16)
# fig.subplots_adjust(top=0.9)
filename = "figure_Normalized_re_comp.png"
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()


tname = f'Normalized Scores with Prior-based sampling using IE Selection'
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=figsize )
fig.tight_layout(pad=3.0)
fig.subplots_adjust(hspace=0.5)

# Plot 1
# ax1.set_title('RE: Hydraulic head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax1.plot(n_h_mc_ie_re, color='#61B2D7', label=r'$MC^{H^{BAL}}$', linewidth=1.5)  
line2, = ax1.plot(n_h_mh_ie_re, color='#D76177', label='$MCMC^{H^{BAL}}$', linewidth=1.5)
ax1.legend(handles=[line1, line2], prop={'size': 10})
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel(r'$D_{KL}/D_{KL}^{REF}$')
# Plot 2
# ax2.set_title('IE: Hydraulic head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax2.plot(n_h_mc_ie_ie, color='#61B2D7', label='$MC^{H^{BAL}}$', linewidth=1.5)  
line2, = ax2.plot(n_h_mh_ie_ie, color='#D76177', label='$MCMC^{H^{BAL}}$', linewidth=1.5)
ax2.legend(handles=[line1, line2], prop={'size': 10})
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel(r'$H/H_{REF}$')
# Plot 3
# ax3.set_title('BME: Hydraulic head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax3.plot(n_h_mc_ie_bme, color='#61B2D7', label='$MC^{H^{BAL}}$', linewidth=1.5)  
line2, = ax3.plot(n_h_mh_ie_bme, color='#D76177', label='$MCMC^{H^{BAL}}$', linewidth=1.5)
ax3.legend(handles=[line1, line2], prop={'size': 10})
ax3.set_xlabel('Number of model runs')
ax3.set_ylabel(r'$BME/BME_{Ref}$')
# Plot 4
# ax4.set_title('RE: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax4.plot(n_c_mc_ie_re, color='#61B2D7', label='$MC^{H^{BAL}}$', linewidth=1.5)  
line2, = ax4.plot(n_c_mh_ie_re, color='#D76177', label='$MCMC^{H^{BAL}}$', linewidth=1.5)
ax4.legend(handles=[line1, line2], prop={'size': 10})
ax4.set_xlabel('Number of model runs')
ax4.set_ylabel(r'$D_{KL}/D_{KL}^{REF}$')
# Plot 5
# ax5.set_title('IE: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax5.plot(n_c_mc_ie_ie, color='#61B2D7', label='$MC^{H^{BAL}}$', linewidth=1.5)  
line2, = ax5.plot(n_c_mh_ie_ie, color='#D76177', label='$MCMC^{H^{BAL}}$', linewidth=1.5)
ax5.legend(handles=[line1, line2], prop={'size': 10})
ax5.set_xlabel('Number of model runs')
ax5.set_ylabel(r'$H/H_{REF}$')

# Plot 6
# ax6.set_title('BME: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax6.plot(n_c_mc_ie_bme, color='#61B2D7', label='$MC^{H^{BAL}}$', linewidth=1.5)  
line2, = ax6.plot(n_c_mh_ie_bme, color='#D76177', label='$MCMC^{H^{BAL}}$', linewidth=1.5)
ax6.legend(handles=[line1, line2], prop={'size': 10})
ax6.set_xlabel('Number of model runs')
ax6.set_ylabel(r'$BME/BME_{Ref}$')
# fig.suptitle(tname, fontname='Arial', fontsize=16)
# fig.subplots_adjust(top=0.9)


tname1 = 'Hydraulic head'
tname2 = 'Concentration'

# Subtitle for the first three subplots
fig.text(0.5, 0.94, tname1,  ha='center',fontsize=12, fontweight='bold', fontname='Arial')

# Subtitle for the last three subplots
fig.text(0.5, 0.48, tname2, ha='center', fontsize=12, fontweight='bold', fontname='Arial')
filename = "figure_Normalized_ie_comp.png"
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()


tname = f'Normalized Scores with Posterior-based sampling using IE Selection'
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=figsize )
fig.tight_layout(pad=3.0)
fig.subplots_adjust(hspace=0.5)

# Plot 1
# ax1.set_title('RE: Hydraulic head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax1.plot(n_h_mc_post_ie_re, color='#61B2D7', label='$MC^{H^{BAL}}$', linewidth=1.5)  
line2, = ax1.plot(n_h_mh_post_ie_re, color='#D76177', label='$MCMC^{H^{BAL}}$', linewidth=1.5)
ax1.legend(handles=[line1, line2], prop={'size': 10})
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel(r'$D_{KL}/D_{KL}^{REF}$')
# Plot 2
# ax2.set_title('IE: Hydraulic head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax2.plot(n_h_mc_post_ie_ie, color='#61B2D7', label='$MC^{H^{BAL}}$', linewidth=1.5)  
line2, = ax2.plot(n_h_mh_post_ie_ie, color='#D76177', label='$MCMC^{H^{BAL}}$', linewidth=1.5)
ax2.legend(handles=[line1, line2], prop={'size': 10})
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel(r'$H/H_{REF}$')
# Plot 3
# ax3.set_title('BME: Hydraulic head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax3.plot(n_h_mc_post_ie_bme, color='#61B2D7', label='$MC^{H^{BAL}}$', linewidth=1.5)  
line2, = ax3.plot(n_h_mh_post_ie_bme, color='#D76177', label='$MCMC^{H^{BAL}}$', linewidth=1.5)
ax3.legend(handles=[line1, line2], prop={'size': 10})
ax3.set_xlabel('Number of model runs')
ax3.set_ylabel(r'$BME/BME_{Ref}$')
# Plot 4
# ax4.set_title('RE: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax4.plot(n_c_mc_post_ie_re, color='#61B2D7', label='$MC^{H^{BAL}}$', linewidth=1.5)  
line2, = ax4.plot(n_c_mh_post_ie_re, color='#D76177', label='$MCMC^{H^{BAL}}$', linewidth=1.5)
ax4.legend(handles=[line1, line2], prop={'size': 10})
ax4.set_xlabel('Number of model runs')
ax4.set_ylabel(r'$D_{KL}/D_{KL}^{REF}$')
# Plot 5
# ax5.set_title('IE: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax5.plot(n_c_mc_post_ie_ie, color='#61B2D7', label='$MC^{H^{BAL}}$', linewidth=1.5)  
line2, = ax5.plot(n_c_mh_post_ie_ie, color='#D76177', label='$MCMC^{H^{BAL}}$', linewidth=1.5)
ax5.legend(handles=[line1, line2], prop={'size': 10})
ax5.set_xlabel('Number of model runs')
ax5.set_ylabel(r'$H/H_{REF}$')
# Plot 6
# ax6.set_title('BME: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax6.plot(n_c_mc_post_ie_bme, color='#61B2D7', label='$MC^{H^{BAL}}$', linewidth=1.5)  
line2, = ax6.plot(n_c_mh_post_ie_bme, color='#D76177', label='$MCMC^{H^{BAL}}$', linewidth=1.5)
ax6.legend(handles=[line1, line2], prop={'size': 10})
ax6.set_xlabel('Number of model runs')
ax6.set_ylabel(r'$BME/BME_{Ref}$')
# fig.suptitle(tname, fontname='Arial', fontsize=16)
# fig.subplots_adjust(top=0.9)


tname1 = 'Hydraulic head'
tname2 = 'Concentration'

# Subtitle for the first three subplots
fig.text(0.5, 0.94, tname1,  ha='center',fontsize=12, fontweight='bold', fontname='Arial')

# Subtitle for the last three subplots
fig.text(0.5, 0.48, tname2, ha='center', fontsize=12, fontweight='bold', fontname='Arial')
filename = "figure_Normalized_post_ie_comp.png"
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()



tname = f'Normalized Scores with Posterior-based sampling using RE Selection'
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=figsize )
fig.tight_layout(pad=3.0)
fig.subplots_adjust(hspace=0.5)

# Plot 1
# ax1.set_title('RE: Hydraulic head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax1.plot(n_h_mc_post_re_re, color='#61B2D7', label='$MC^{D_{KL}^{BAL}}$', linewidth=1.5)  
line2, = ax1.plot(n_h_mh_post_re_re, color='#D76177', label='$MCMC^{D_{KL}^{BAL}}$', linewidth=1.5)
ax1.legend(handles=[line1, line2], prop={'size': 10})
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel(r'$D_{KL}/D_{KL}^{REF}$')
# Plot 2
# ax2.set_title('IE: Hydraulic head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax2.plot(n_h_mc_post_re_ie, color='#61B2D7', label='$MC^{D_{KL}^{BAL}}$', linewidth=1.5)  
line2, = ax2.plot(n_h_mh_post_re_ie, color='#D76177', label='$MCMC^{D_{KL}^{BAL}}$', linewidth=1.5)
ax2.legend(handles=[line1, line2], prop={'size': 10})
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel(r'$H/H_{REF}$')
# Plot 3
# ax3.set_title('BME: Hydraulic head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax3.plot(n_h_mc_post_re_bme, color='#61B2D7', label='$MC^{D_{KL}^{BAL}}$', linewidth=1.5)  
line2, = ax3.plot(n_h_mh_post_re_bme, color='#D76177', label='$MCMC^{D_{KL}^{BAL}}$', linewidth=1.5)
ax3.legend(handles=[line1, line2], prop={'size': 10})
ax3.set_xlabel('Number of model runs')
ax3.set_ylabel(r'$BME/BME_{Ref}$')
# Plot 4
# ax4.set_title('RE: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax4.plot(n_c_mc_post_re_re, color='#61B2D7', label='$MC^{D_{KL}^{BAL}}$', linewidth=1.5)  
line2, = ax4.plot(n_c_mh_post_re_re, color='#D76177', label='$MCMC^{D_{KL}^{BAL}}$', linewidth=1.5)
ax4.legend(handles=[line1, line2], loc='lower right', prop={'size': 10})
ax4.set_xlabel('Number of model runs')
ax4.set_ylabel(r'$D_{KL}/D_{KL}^{REF}$')
# Plot 5
# ax5.set_title('IE: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax5.plot(n_c_mc_post_re_ie, color='#61B2D7', label='$MC^{D_{KL}^{BAL}}$', linewidth=1.5)  
line2, = ax5.plot(n_c_mh_post_re_ie, color='#D76177', label='$MCMC^{D_{KL}^{BAL}}$', linewidth=1.5)
ax5.legend(handles=[line1, line2], prop={'size': 10})
ax5.set_xlabel('Number of model runs')
ax5.set_ylabel(r'$H/H_{REF}$')
# Plot 6
# ax6.set_title('BME: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax6.plot(n_c_mc_post_re_bme, color='#61B2D7', label='$MC^{D_{KL}^{BAL}}$', linewidth=1.5)  
line2, = ax6.plot(n_c_mh_post_re_bme, color='#D76177', label='$MCMC^{D_{KL}^{BAL}}$', linewidth=1.5)
ax6.legend(handles=[line1, line2], prop={'size': 10})
ax6.set_xlabel('Number of model runs')
ax6.set_ylabel(r'$BME/BME_{Ref}$')
# fig.suptitle(tname, fontname='Arial', fontsize=16)
# fig.subplots_adjust(top=0.9)

tname1 = 'Hydraulic head'
tname2 = 'Concentration'
# Subtitle for the first three subplots
fig.text(0.5, 0.94, tname1,  ha='center',fontsize=12, fontweight='bold', fontname='Arial')

# Subtitle for the last three subplots
fig.text(0.5, 0.48, tname2, ha='center', fontsize=12, fontweight='bold', fontname='Arial')
filename = "figure_Normalized_post_re_comp.png"
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()




tname = f'Normalized Scores with Posterior-based sampling using BME Selection'
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=figsize )
fig.tight_layout(pad=3.0)
fig.subplots_adjust(hspace=0.5)

# Plot 1
# ax1.set_title('RE: Hydraulic head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax1.plot(n_h_mc_post_bme_re, color='#61B2D7', label='$MC^{BME^{BAL}}$', linewidth=1.5)  
line2, = ax1.plot(n_h_mh_post_bme_re, color='#D76177', label='$MCMC^{BME^{BAL}}$', linewidth=1.5)
ax1.legend(handles=[line1, line2], loc='lower right', prop={'size': 10})
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel(r'$D_{KL}/D_{KL}^{REF}$')
# Plot 2
# ax2.set_title('IE: Hydraulic head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax2.plot(n_h_mc_post_bme_ie, color='#61B2D7', label='$MC^{BME^{BAL}}$', linewidth=1.5)  
line2, = ax2.plot(n_h_mh_post_bme_ie, color='#D76177', label='$MCMC^{BME^{BAL}}$', linewidth=1.5)
ax2.legend(handles=[line1, line2], prop={'size': 10})
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel(r'$H/H_{REF}$')
# Plot 3
# ax3.set_title('BME: Hydraulic head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax3.plot(n_h_mc_post_bme_bme, color='#61B2D7', label='$MC^{BME^{BAL}}$', linewidth=1.5)  
line2, = ax3.plot(n_h_mh_post_bme_bme, color='#D76177', label='$MCMC^{BME^{BAL}}$', linewidth=1.5)
ax3.legend(handles=[line1, line2], prop={'size': 10})
ax3.set_xlabel('Number of model runs')
ax3.set_ylabel(r'$BME/BME_{Ref}$')
# Plot 4
# ax4.set_title('RE: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax4.plot(n_c_mc_post_bme_re, color='#61B2D7', label='$MC^{BME^{BAL}}$', linewidth=1.5)  
line2, = ax4.plot(n_c_mh_post_bme_re, color='#D76177', label='$MCMC^{BME^{BAL}}$', linewidth=1.5)
ax4.legend(handles=[line1, line2], prop={'size': 10})
ax4.set_xlabel('Number of model runs')
ax4.set_ylabel(r'$D_{KL}/D_{KL}^{REF}$')
# Plot 5
# ax5.set_title('IE: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax5.plot(n_c_mc_post_bme_ie, color='#61B2D7', label='$MC^{BME^{BAL}}$', linewidth=1.5)  
line2, = ax5.plot(n_c_mh_post_bme_ie, color='#D76177', label='$MCMC^{BME^{BAL}}$', linewidth=1.5)
ax5.legend(handles=[line1, line2], prop={'size': 10})
ax5.set_xlabel('Number of model runs')
ax5.set_ylabel(r'$H/H_{REF}$')
# Plot 6
# ax6.set_title('BME: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax6.plot(n_c_mc_post_bme_bme, color='#61B2D7', label='$MC^{BME^{BAL}}$', linewidth=1.5)  
line2, = ax6.plot(n_c_mh_post_bme_bme, color='#D76177', label='$MCMC^{BME^{BAL}}$', linewidth=1.5)
ax6.legend(handles=[line1, line2], prop={'size': 10})
ax6.set_xlabel('Number of model runs')
ax6.set_ylabel(r'$BME/BME_{Ref}$')


tname1 = 'Hydraulic head'
tname2 = 'Concentration'

# Subtitle for the first three subplots
fig.text(0.5, 0.94, tname1,  ha='center',fontsize=12, fontweight='bold', fontname='Arial')

# Subtitle for the last three subplots
fig.text(0.5, 0.48, tname2, ha='center', fontsize=12, fontweight='bold', fontname='Arial')
# fig.suptitle(tname, fontname='Arial', fontsize=16)
# fig.subplots_adjust(top=0.9)
filename = "figure_Normalized_post_bme_comp.png"
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()



cp_prior_mc_bme = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/BME/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]
cp_prior_mc_re = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/RE/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]
cp_prior_mc_ie = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/IE/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]

cp_post_mc_bme = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/post/BME/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]
cp_post_mc_re = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/post/RE/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]
cp_post_mc_ie = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/post/IE/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]

cp_prior_mh_bme = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/BME/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]
cp_prior_mh_re = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/RE/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]
cp_prior_mh_ie = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/IE/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]

cp_post_mh_bme = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/post/BME/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]
cp_post_mh_re = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/post/RE/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]
cp_post_mh_ie = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/post/IE/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]

tname = f'Training points for Zone 1 with Prior-based sampling using diffrent Selection'
fig, ((ax1, ax2, ax3)) = plt.subplots(3, 1, figsize=figsize )
fig.tight_layout(pad=3.0)
fig.subplots_adjust(hspace=0.5)

# Plot 1
ax1.set_title('RE', fontname='Arial', fontsize=12, weight='bold')
line1, = ax1.plot(cp_prior_mc_re, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax1.plot(cp_prior_mh_re, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax1.legend(handles=[line1, line2], prop={'size': 9})
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel(r'$Log(HK)$')

# Plot 2
ax2.set_title('IE', fontname='Arial', fontsize=12, weight='bold')
line1, = ax2.plot(cp_prior_mc_ie, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax2.plot(cp_prior_mh_ie, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax2.legend(handles=[line1, line2], prop={'size': 9})
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel(r'$Log(HK)$')
# Plot 3
ax3.set_title('BME', fontname='Arial', fontsize=12, weight='bold')
line1, = ax3.plot(cp_prior_mc_bme, color='#61B2D7', label='MC', linewidth=1.5)  
line2, = ax3.plot(cp_prior_mh_bme, color='#D76177', label='MCMC-MH', linewidth=1.5)
ax3.legend(handles=[line1, line2], prop={'size': 9})
ax3.set_xlabel('Number of model runs')
ax3.set_ylabel(r'$Log(HK)$')

# fig.suptitle(tname, fontname='Arial', fontsize=16)
# fig.subplots_adjust(top=0.9)
filename = "figure_cp_prior_comp.png"
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()

# cp_prior_mc_bme = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/BME/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]
# cp_prior_mc_re = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/RE/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]
# cp_prior_mc_ie = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MC/prior/IE/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]


# cp_prior_mh_bme = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/BME/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]
# cp_prior_mh_re = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/RE/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]
# cp_prior_mh_ie = np.array(pd.read_csv(('C:/Users/tian/Desktop/gp/backup/outputBAL30_270/MH/prior/IE/collocation_points.csv'), dtype=np.float64, header = 0, index_col = 0))[30:-1,0]









# tname = f'Training points for Zone 1 with Posterior-based sampling using diffrent Selection'
# fig, ((ax1, ax2, ax3)) = plt.subplots(3, 1, figsize=figsize )
# fig.tight_layout(pad=3.0)
# fig.subplots_adjust(hspace=0.5)

# # Plot 1
# ax1.set_title('RE', fontname='Arial', fontsize=12, weight='bold')
# line1, = ax1.plot(cp_post_mc_re, color='#61B2D7', label='MC', linewidth=1.5)  
# line2, = ax1.plot(cp_post_mh_re, color='#D76177', label='MCMC-MH', linewidth=1.5)
# ax1.legend(handles=[line1, line2], prop={'size': 9})
# ax1.set_xlabel('Number of model runs')
# ax1.set_ylabel(r'$Log(HK)$')
# # Plot 2
# ax2.set_title('IE', fontname='Arial', fontsize=12, weight='bold')
# line1, = ax2.plot(cp_post_mc_ie, color='#61B2D7', label='MC', linewidth=1.5)  
# line2, = ax2.plot(cp_post_mh_ie, color='#D76177', label='MCMC-MH', linewidth=1.5)
# ax2.legend(handles=[line1, line2], prop={'size': 9})
# ax2.set_xlabel('Number of model runs')
# ax2.set_ylabel(r'$Log(HK)$')
# # Plot 3
# ax3.set_title('BME', fontname='Arial', fontsize=12, weight='bold')
# line1, = ax3.plot(cp_post_mc_bme, color='#61B2D7', label='MC', linewidth=1.5)  
# line2, = ax3.plot(cp_post_mh_bme, color='#D76177', label='MCMC-MH', linewidth=1.5)
# ax3.legend(handles=[line1, line2], prop={'size': 9})
# ax3.set_xlabel('Number of model runs')
# ax3.set_ylabel(r'$Log(HK)$')

# # fig.suptitle(tname, fontname='Arial', fontsize=16)
# # fig.subplots_adjust(top=0.9)
# filename = "figure_cp_post_comp.png"
# plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
# plt.show()



# tname = f'Training points for Zone 1 with Prior-based sampling using diffrent Selection'
# fig, ((ax1, ax2, ax3)) = plt.subplots(3, 1, figsize=figsize )
# fig.tight_layout(pad=3.0)
# fig.subplots_adjust(hspace=0.5)

# # Plot 1
# ax1.set_title('RE', fontname='Arial', fontsize=12, weight='bold')
# line1, = ax1.plot(cp_prior_mc_re, color='#61B2D7', label='MC', linewidth=1.5)  
# line2, = ax1.plot(cp_prior_mh_re, color='#D76177', label='MCMC-MH', linewidth=1.5)
# ax1.legend(handles=[line1, line2], prop={'size': 9})
# ax1.set_xlabel('Number of model runs')
# ax1.set_ylabel(r'$Log(HK)$')
# # Plot 2
# ax2.set_title('IE', fontname='Arial', fontsize=12, weight='bold')
# line1, = ax2.plot(cp_prior_mc_ie, color='#61B2D7', label='MC', linewidth=1.5)  
# line2, = ax2.plot(cp_prior_mh_ie, color='#D76177', label='MCMC-MH', linewidth=1.5)
# ax2.legend(handles=[line1, line2], prop={'size': 9})
# ax2.set_xlabel('Number of model runs')
# ax2.set_ylabel(r'$Log(HK)$')
# # Plot 3
# ax3.set_title('BME', fontname='Arial', fontsize=12, weight='bold')
# line1, = ax3.plot(cp_prior_mc_bme, color='#61B2D7', label='MC', linewidth=1.5)  
# line2, = ax3.plot(cp_prior_mh_bme, color='#D76177', label='MCMC-MH', linewidth=1.5)
# ax3.legend(handles=[line1, line2], prop={'size': 9})
# ax3.set_xlabel('Number of model runs')
# ax3.set_ylabel(r'$Log(HK)$')

# # fig.suptitle(tname, fontname='Arial', fontsize=16)
# # fig.subplots_adjust(top=0.9)
# filename = "figure_cp_prior_comp.png"
# plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
# plt.show()






tname = f'Training points for Zone 1 sampling MC'
fig, ((ax1, ax2)) = plt.subplots(2, 1, figsize=figsize )
fig.tight_layout(pad=3.0)
fig.subplots_adjust(hspace=0.5)

# Plot 1
ax1.set_title('Random Sampling', fontname='Arial', fontsize=12, weight='bold')
line1, = ax1.plot(cp_prior_mc_re, color=cre, label='$RE^{BAL}$', linewidth=1.5)  
line2, = ax1.plot(cp_prior_mc_bme, color=cbme, label='$BME^{BAL}$', linewidth=1.5)
line3, = ax1.plot(cp_prior_mc_ie, color=cie, label='$IE^{BAL}$', linewidth=1.5)
ax1.legend(handles=[line1, line2, line3], prop={'size': 9})
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel(r'$Log(HK)$')
# Plot 2
ax2.set_title('MCMC-based Sampling', fontname='Arial', fontsize=12, weight='bold')
line1, = ax2.plot(cp_prior_mh_re, color=cre, label='$RE^{BAL}$', linewidth=1.5)  
line2, = ax2.plot(cp_prior_mh_bme, color=cbme, label='$BME^{BAL}$', linewidth=1.5)
line3, = ax2.plot(cp_prior_mh_ie, color=cie, label='$IE^{BAL}$', linewidth=1.5)
ax2.legend(handles=[line1, line2, line3], prop={'size': 9})
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel(r'$Log(HK)$')
# fig.suptitle(tname, fontname='Arial', fontsize=16)
# fig.subplots_adjust(top=0.9)
filename = "figure_cp_prior_mcmh.png"
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()



tname = f'Training points for Zone 1 sampling MH'
fig, ((ax1, ax2)) = plt.subplots(2, 1, figsize=figsize )
fig.tight_layout(pad=3.0)
fig.subplots_adjust(hspace=0.5)

# Plot 1
ax1.set_title('MC-based Sampling', fontname='Arial', fontsize=12, weight='bold')
line1, = ax1.plot(cp_post_mc_re, color=cre, label='$RE^{BAL}$', linewidth=1.5)  
line2, = ax1.plot(cp_post_mc_bme, color=cbme, label='$BME^{BAL}$', linewidth=1.5)
line3, = ax1.plot(cp_post_mc_ie, color=cie, label='$IE^{BAL}$', linewidth=1.5)
ax1.legend(handles=[line1, line2, line3], prop={'size': 9})
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel(r'$Log(HK)$')
# Plot 2
ax2.set_title('MCMC-based Sampling', fontname='Arial', fontsize=12, weight='bold')
line1, = ax2.plot(cp_post_mh_re, color=cre, label='$RE^{BAL}$', linewidth=1.5)  
line2, = ax2.plot(cp_post_mh_bme, color=cbme, label='$BME^{BAL}$', linewidth=1.5)
line3, = ax2.plot(cp_post_mh_ie, color=cie, label='$IE^{BAL}$', linewidth=1.5)
ax2.legend(handles=[line1, line2, line3], prop={'size': 9})
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel(r'$Log(HK)$')
# fig.suptitle(tname, fontname='Arial', fontsize=16)
# fig.subplots_adjust(top=0.9)
filename = "figure_cp_post_mcmh.png"
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()





tname = f'Training points for Zone 1 sampling MC'

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize )
fig.tight_layout(pad=3.0)
fig.subplots_adjust(hspace=0.5)

# Plot 1
ax1.set_title('MC-based Sampling', fontname='Arial', fontsize=12, weight='bold')
ax1.hist([cp_post_mc_re, cp_post_mc_bme, cp_post_mc_ie], bins=30, color=[cre, cbme, cie], label=['$RE^{BAL}$', '$BME^{BAL}$', '$IE^{BAL}$'])
ax1.legend(prop={'size': 9})
ax1.set_xlabel(r'$Log(HK)$')
# Plot 2
ax2.set_title('MCMC-based Sampling', fontname='Arial', fontsize=12, weight='bold')
ax2.hist([cp_post_mh_re, cp_post_mh_bme, cp_post_mh_ie], bins=30, color=[cre, cbme, cie], label=['$RE^{BAL}$', '$BME^{BAL}$', '$IE^{BAL}$'])
ax2.legend(prop={'size': 9})
ax2.set_xlabel(r'$Log(HK)$')
# fig.suptitle(tname, fontname='Arial', fontsize=16)
# fig.subplots_adjust(top=0.9)
filename = "figure_cp_post_mcmh_his.png"
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()



tname = f'Training points for Zone 1'

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize )
fig.tight_layout(pad=3.0)
fig.subplots_adjust(hspace=0.5)

# Plot 1 (prior)
ax1.set_title('Random Sampling', fontname='Arial', fontsize=12, weight='bold')
ax1.hist([cp_prior_mc_re, cp_prior_mc_bme, cp_prior_mc_ie], bins=30, color=[cre, cbme, cie], label=['$RE^{BAL}$', '$BME^{BAL}$', '$IE^{BAL}$'])
ax1.legend(prop={'size': 9})
ax1.set_xlabel(r'$Log(HK)$')
# Plot 2 (posterior)
ax2.set_title('MCMC-based Sampling', fontname='Arial', fontsize=12, weight='bold')
ax2.hist([cp_prior_mh_re, cp_prior_mh_bme, cp_prior_mh_ie], bins=30, color=[cre, cbme, cie], label=['$RE^{BAL}$', '$BME^{BAL}$', '$IE^{BAL}$'])
ax2.legend(prop={'size': 9})
ax2.set_xlabel(r'$Log(HK)$')
# fig.suptitle(tname, fontname='Arial', fontsize=16)
# fig.subplots_adjust(top=0.9)
filename = "figure_cp_prior_mcmh_his.png"
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()




figsize = (12, 7.416)
tname = f'Normalized Scores with Posterior-based sampling using IE Selection'
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=figsize )
fig.tight_layout(pad=3.0)
fig.subplots_adjust(hspace=0.5)

# Plot 1
# ax1.set_title('RE: Hydraulic head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax1.plot(n_h_mc_post_ie_re, color=cbme, label='$MC^{H^{BAL}}_{post}$', linewidth=1.5)  
line2, = ax1.plot(n_h_mh_post_ie_re, color=cre, label='$MCMC^{H^{BAL}}_{post}$', linewidth=1.5)
line3, = ax1.plot(n_h_mc_ie_re, color=cie, label='$MC^{H^{BAL}}_{prior}$', linewidth=1.5)
ax1.legend(handles=[line1, line2, line3], prop={'size': 8})
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel(r'$D_{KL}/D_{KL}^{REF}$')
# Plot 2
# ax2.set_title('IE: Hydraulic head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax2.plot(n_h_mc_post_ie_ie, color=cbme, label='$MC^{H^{BAL}}_{post}$', linewidth=1.5)  
line2, = ax2.plot(n_h_mh_post_ie_ie, color=cre, label='$MCMC^{H^{BAL}}_{post}$', linewidth=1.5)
line3, = ax2.plot(n_h_mc_ie_ie, color=cie, label='$MC^{H^{BAL}}_{prior}$', linewidth=1.5)
ax2.legend(handles=[line1, line2, line3], prop={'size': 8})
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel(r'$H/H_{REF}$')
# Plot 3
# ax3.set_title('BME: Hydraulic head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax3.plot(n_h_mc_post_ie_bme, color=cbme, label='$MC^{H^{BAL}}_{post}$', linewidth=1.5)  
line2, = ax3.plot(n_h_mh_post_ie_bme, color=cre, label='$MCMC^{H^{BAL}}_{post}$', linewidth=1.5)
line3, = ax3.plot(n_h_mc_ie_bme, color=cie, label='$MC^{H^{BAL}}_{prior}$', linewidth=1.5)
ax3.legend(handles=[line1, line2, line3], prop={'size': 8})
ax3.set_xlabel('Number of model runs')
ax3.set_ylabel(r'$BME/BME_{Ref}$')
# Plot 4
# ax4.set_title('RE: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax4.plot(n_c_mc_post_ie_re, color=cbme, label='$MC^{H^{BAL}}_{post}$', linewidth=1.5)  
line2, = ax4.plot(n_c_mh_post_ie_re, color=cre, label='$MCMC^{H^{BAL}}_{post}$', linewidth=1.5)
line3, = ax4.plot(n_c_mc_ie_re, color=cie, label='$MC^{H^{BAL}}_{prior}$', linewidth=1.5)
ax4.legend(handles=[line1, line2, line3], prop={'size': 8})
ax4.set_xlabel('Number of model runs')
ax4.set_ylabel(r'$D_{KL}/D_{KL}^{REF}$')
# Plot 5
# ax5.set_title('IE: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax5.plot(n_c_mc_post_ie_ie, color=cbme, label='$MC^{H^{BAL}}_{post}$', linewidth=1.5)  
line2, = ax5.plot(n_c_mh_post_ie_ie, color=cre, label='$MCMC^{H^{BAL}}_{post}$', linewidth=1.5)
line3, = ax5.plot(n_c_mc_ie_ie, color=cie, label='$MC^{H^{BAL}}_{prior}$', linewidth=1.5)
ax5.legend(handles=[line1, line2, line3], prop={'size':8})
ax5.set_xlabel('Number of model runs')
ax5.set_ylabel(r'$H/H_{REF}$')
# Plot 6
# ax6.set_title('BME: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax6.plot(n_c_mc_post_ie_bme, color=cbme, label='$MC^{H^{BAL}}_{post}$', linewidth=1.5)  
line2, = ax6.plot(n_c_mh_post_ie_bme, color=cre, label='$MCMC^{H^{BAL}}_{post}$', linewidth=1.5)
line3, = ax6.plot(n_c_mc_ie_bme, color=cie, label='$MC^{H^{BAL}}_{prior}$', linewidth=1.5)
ax6.legend(handles=[line1, line2, line3], prop={'size': 8})
ax6.set_xlabel('Number of model runs')
ax6.set_ylabel(r'$BME/BME_{Ref}$')
# fig.suptitle(tname, fontname='Arial', fontsize=16)
# fig.subplots_adjust(top=0.9)


tname1 = 'Hydraulic head'
tname2 = 'Concentration'

# Subtitle for the first three subplots
fig.text(0.5, 0.94, tname1,  ha='center',fontsize=12, fontweight='bold', fontname='Arial')

# Subtitle for the last three subplots
fig.text(0.5, 0.48, tname2, ha='center', fontsize=12, fontweight='bold', fontname='Arial')
filename = "figure_Normalized_post_ie_bcomp.png"
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()



tname = f'Normalized Scores with Posterior-based sampling using RE Selection'
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=figsize )
fig.tight_layout(pad=3.0)
fig.subplots_adjust(hspace=0.5)

# Plot 1
# ax1.set_title('RE: Hydraulic head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax1.plot(n_h_mc_post_re_re, color=cbme, label='$MC^{D_{KL}^{BAL}}_{post}$', linewidth=1.5)  
line2, = ax1.plot(n_h_mh_post_re_re, color=cre, label='$MCMC^{D_{KL}^{BAL}}_{post}$', linewidth=1.5)
line3, = ax1.plot(n_h_mc_re_re, color=cie, label='$MC^{D_{KL}^{BAL}}_{prior}$', linewidth=1.5)
ax1.legend(handles=[line1, line2, line3], prop={'size': 8})
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel(r'$D_{KL}/D_{KL}^{REF}$')
# Plot 2
# ax2.set_title('IE: Hydraulic head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax2.plot(n_h_mc_post_re_ie, color=cbme, label='$MC^{D_{KL}^{BAL}}_{post}$', linewidth=1.5)  
line2, = ax2.plot(n_h_mh_post_re_ie, color=cre, label='$MCMC^{D_{KL}^{BAL}}_{post}$', linewidth=1.5)
line3, = ax2.plot(n_h_mc_re_ie, color=cie, label='$MC^{D_{KL}^{BAL}}_{prior}$', linewidth=1.5)
ax2.legend(handles=[line1, line2, line3], prop={'size': 8})
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel(r'$H/H_{REF}$')
# Plot 3
# ax3.set_title('BME: Hydraulic head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax3.plot(n_h_mc_post_re_bme, color=cbme, label='$MC^{D_{KL}^{BAL}}_{post}$', linewidth=1.5)  
line2, = ax3.plot(n_h_mh_post_re_bme, color=cre, label='$MCMC^{D_{KL}^{BAL}}_{post}$', linewidth=1.5)
line3, = ax3.plot(n_h_mc_re_bme, color=cie, label='$MC^{D_{KL}^{BAL}}_{prior}$', linewidth=1.5)
ax3.legend(handles=[line1, line2, line3], prop={'size': 7.5})
ax3.set_xlabel('Number of model runs')
ax3.set_ylabel(r'$BME/BME_{Ref}$')
# Plot 4
# ax4.set_title('RE: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax4.plot(n_c_mc_post_re_re, color=cbme, label='$MC^{D_{KL}^{BAL}}_{post}$', linewidth=1.5)  
line2, = ax4.plot(n_c_mh_post_re_re, color=cre, label='$MCMC^{D_{KL}^{BAL}}_{post}$', linewidth=1.5)
line3, = ax4.plot(n_c_mc_re_re, color=cie, label='$MC^{D_{KL}^{BAL}}_{prior}$', linewidth=1.5)
ax4.legend(handles=[line1, line2, line3], prop={'size': 8})
ax4.set_xlabel('Number of model runs')
ax4.set_ylabel(r'$D_{KL}/D_{KL}^{REF}$')
# Plot 5
# ax5.set_title('IE: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax5.plot(n_c_mc_post_re_ie, color=cbme, label='$MC^{D_{KL}^{BAL}}_{post}$', linewidth=1.5)  
line2, = ax5.plot(n_c_mh_post_re_ie, color=cre, label='$MCMC^{D_{KL}^{BAL}}_{post}$', linewidth=1.5)
line3, = ax5.plot(n_c_mc_re_ie, color=cie, label='$MC^{D_{KL}^{BAL}}_{prior}$', linewidth=1.5)
ax5.legend(handles=[line1, line2, line3], prop={'size': 8})
ax5.set_xlabel('Number of model runs')
ax5.set_ylabel(r'$H/H_{REF}$')
# Plot 6
# ax6.set_title('BME: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax6.plot(n_c_mc_post_re_bme, color=cbme, label='$MC^{D_{KL}^{BAL}}_{post}$', linewidth=1.5)  
line2, = ax6.plot(n_c_mh_post_re_bme, color=cre, label='$MCMC^{D_{KL}^{BAL}}_{post}$', linewidth=1.5)
line3, = ax6.plot(n_c_mc_re_bme, color=cie, label='$MC^{D_{KL}^{BAL}}_{prior}$', linewidth=1.5)
ax6.legend(handles=[line1, line2, line3], prop={'size': 8})
ax6.set_xlabel('Number of model runs')
ax6.set_ylabel(r'$BME/BME_{Ref}$')
# fig.suptitle(tname, fontname='Arial', fontsize=16)
# fig.subplots_adjust(top=0.9)


tname1 = 'Hydraulic head'
tname2 = 'Concentration'

# Subtitle for the first three subplots
fig.text(0.5, 0.94, tname1,  ha='center',fontsize=12, fontweight='bold', fontname='Arial')

# Subtitle for the last three subplots
fig.text(0.5, 0.48, tname2, ha='center', fontsize=12, fontweight='bold', fontname='Arial')
filename = "figure_Normalized_post_re_bcomp.png"
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()


tname = f'Normalized Scores with Posterior-based sampling using BME Selection'
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=figsize )
fig.tight_layout(pad=3.0)
fig.subplots_adjust(hspace=0.5)

# Plot 1
# ax1.set_title('RE: Hydraulic head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax1.plot(n_h_mc_post_bme_re, color=cbme, label='$MC^{BME^{BAL}}_{post}$', linewidth=1.5)  
line2, = ax1.plot(n_h_mh_post_bme_re, color=cre, label='$MCMC^{BME^{BAL}}_{post}$', linewidth=1.5)
line3, = ax1.plot(n_h_mc_re_bme, color=cie, label='$MC^{BME^{BAL}}_{prior}$', linewidth=1.5)
ax1.legend(handles=[line1, line2, line3], loc= 2,prop={'size': 8})
ax1.set_xlabel('Number of model runs')
ax1.set_ylabel(r'$D_{KL}/D_{KL}^{REF}$')
# Plot 2
# ax2.set_title('IE: Hydraulic head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax2.plot(n_h_mc_post_bme_ie, color=cbme, label='$MC^{BME^{BAL}}_{post}$', linewidth=1.5)  
line2, = ax2.plot(n_h_mh_post_bme_ie, color=cre, label='$MCMC^{BME^{BAL}}_{post}$', linewidth=1.5)
line3, = ax2.plot(n_h_mc_bme_ie, color=cie, label='$MC^{BME^{BAL}}_{prior}$', linewidth=1.5)
ax2.legend(handles=[line1, line2, line3], prop={'size': 8})
ax2.set_xlabel('Number of model runs')
ax2.set_ylabel(r'$H/H_{REF}$')
# Plot 3
# ax3.set_title('BME: Hydraulic head', fontname='Arial', fontsize=12, weight='bold')
line1, = ax3.plot(n_h_mc_post_bme_bme, color=cbme, label='$MC^{BME^{BAL}}_{post}$', linewidth=1.5)  
line2, = ax3.plot(n_h_mh_post_bme_bme, color=cre, label='$MCMC^{BME^{BAL}}_{post}$', linewidth=1.5)
line3, = ax3.plot(n_h_mc_bme_bme, color=cie, label='$MC^{BME^{BAL}}_{prior}$', linewidth=1.5)
ax3.legend(handles=[line1, line2, line3], prop={'size': 8})
ax3.set_xlabel('Number of model runs')
ax3.set_ylabel(r'$BME/BME_{Ref}$')
# Plot 4
# ax4.set_title('RE: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax4.plot(n_c_mc_post_bme_re, color=cbme, label='$MC^{BME^{BAL}}_{post}$', linewidth=1.5)  
line2, = ax4.plot(n_c_mh_post_bme_re, color=cre, label='$MCMC^{BME^{BAL}}_{post}$', linewidth=1.5)
line3, = ax4.plot(n_c_mc_re_bme, color=cie, label='$MC^{BME^{BAL}}_{prior}$', linewidth=1.5)
ax4.legend(handles=[line1, line2, line3], prop={'size': 8})
ax4.set_xlabel('Number of model runs')
ax4.set_ylabel(r'$D_{KL}/D_{KL}^{REF}$')
# Plot 5
# ax5.set_title('IE: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax5.plot(n_c_mc_post_bme_ie, color=cbme, label='$MC^{BME^{BAL}}_{post}$', linewidth=1.5)  
line2, = ax5.plot(n_c_mh_post_bme_ie, color=cre, label='$MCMC^{BME^{BAL}}_{post}$', linewidth=1.5)
line3, = ax5.plot(n_c_mc_bme_ie, color=cie, label='$MC^{BME^{BAL}}_{prior}$', linewidth=1.5)
ax5.legend(handles=[line1, line2, line3], prop={'size': 8})
ax5.set_xlabel('Number of model runs')
ax5.set_ylabel(r'$H/H_{REF}$')
# Plot 6
# ax6.set_title('BME: Concentration', fontname='Arial', fontsize=12, weight='bold')
line1, = ax6.plot(n_c_mc_post_bme_bme, color=cbme, label='$MC^{BME^{BAL}}_{post}$', linewidth=1.5)  
line2, = ax6.plot(n_c_mh_post_bme_bme, color=cre, label='$MCMC^{BME^{BAL}}_{post}$', linewidth=1.5)
line3, = ax6.plot(n_c_mc_bme_bme, color=cie, label='$MC^{BME^{BAL}}_{prior}$', linewidth=1.5)
ax6.legend(handles=[line1, line2, line3], prop={'size': 8})
ax6.set_xlabel('Number of model runs')
ax6.set_ylabel(r'$BME/BME_{Ref}$')
# fig.suptitle(tname, fontname='Arial', fontsize=16)
# fig.subplots_adjust(top=0.9)


tname1 = 'Hydraulic head'
tname2 = 'Concentration'

# Subtitle for the first three subplots
fig.text(0.5, 0.94, tname1,  ha='center',fontsize=12, fontweight='bold', fontname='Arial')

# Subtitle for the last three subplots
fig.text(0.5, 0.48, tname2, ha='center', fontsize=12, fontweight='bold', fontname='Arial')
filename = "figure_Normalized_post_bme_bcomp.png"
plt.savefig((model_output_path + f'/plots/{filename}'), bbox_inches="tight")
plt.show()
