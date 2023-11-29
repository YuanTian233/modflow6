# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 01:13:53 2023

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
# import chaospy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import WhiteKernel



import numpy as np
import scipy.stats as stats

model_output_path = 'C:/Users/tian/Desktop/gp/testbal_mc'
if os.path.exists(model_output_path + '/output'):
    print('path exists')
else:
    os.mkdir(model_output_path + '/output')

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
covariance_matrices = np.random.random(size=(20,20))
IE = 0.5 * (np.log(2 * math.pi * math.exp(1)**6) + np.log(covariance_matrices))



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
# cmap = plt.get_cmap('GnBu', 6)
# mod_zones01 = c#[0 : nrow, 0 : ncol]
# im3 = ax1.imshow(mod_zones01, cmap=cmap
#                  #['#F309AC', '#F30937', '#09ACF3', '#F3C509', '#ACF309','#9703FF']) 
#                  )
# ax1.set_title("Zone Distribution")
# bounds = [1, 2, 3, 4, 5, 6]
# cbar = fig.colorbar(im3, 
#                     ax=ax1,  
#                     ticks=bounds, 
#                     spacing='proportional'
#                     )


# ax1.scatter(point_loc[:, 1], point_loc[:, 0], marker='o',c='#112F2C')        
# plt.show()        






# from matplotlib.colors import ListedColormap

# Create a 50x50 grid
grid_size = 50
grid = np.zeros((grid_size, grid_size))  # Initialize the grid with zeros

# Set the first column (column index 0) to red (value 1.0)
grid[:, 0] = 1.0
grid[:, -1] = 1.0
grid[20:31, 0] = 0.5
# Create a custom colormap with red for value 1.0 and white for value 0.0
colors = ['white', 'red','blue']
cmap = ListedColormap(colors)

# Create a figure and axis for the map
fig, ax = plt.subplots()

# Display the grid as a map with the custom colormap
im = ax.imshow(grid, cmap=cmap, interpolation='none', aspect='auto', extent=[0, grid_size, 0, grid_size])

# Add a color bar for reference


# Set axis labels and title
ax.set_xlabel('No Flow')
ax.set_ylabel('H = 1 m')
ax.set_title('No Flow')
ax_right = ax.twinx()
ax_right.set_yticks(np.arange(0, grid_size+1, 10))
ax_right.set_ylabel('H = 0 m')
# Show the map
plt.show()


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
# fig, ax = plt.subplots()

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
# plt.show()