# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 14:48:38 2023

@author: tian
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

zones_vec50x50 = pd.read_csv('C:/Users/tian/Desktop/testzones/zone_distribution.csv', dtype=np.float64)

zones_vec = np.array(zones_vec50x50, dtype = float)






ax = plt.subplot()
cmap = mpl.cm.viridis
bounds = [1, 2, 3, 4, 5, 6]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
im = ax.imshow(zones_vec,cmap=cmap, norm=norm)


plt.title("Log(K) Field")


plt.colorbar(im)
# plt.scatter(point_loc[:, 1], point_loc[:, 0], marker='o',edgecolors='w')


plt.show()

