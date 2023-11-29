# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 16:48:07 2023

@author: tian
"""
import math
import pandas as pd
import numpy as np





n_zones = 6 
n_reali =  1
gm_name = "exponential"
gm_mean = math.log(1e-5)  # ln(K) average --> constant
gm_var = 1.0  # ln(K) variance --> constant
gm_correl_length = np.array([10.0, 10.0])  # correlation length in [x, y] for geostatistical model
parameter_sets = np.random.normal(loc=gm_mean, 
                                  scale=gm_var,
                                  size=(n_reali, n_zones)
                                  )



from gwf50x50 import hkfields
from gwf50x50 import smooth
from gwf50x50 import rungwfmodle
from gwf50x50 import plotgwm
zones_vec50x50 = pd.read_csv('C:/Users/tian/Desktop/testzones/zone_distribution.csv', dtype=np.float64)
zones_vec = np.array(zones_vec50x50, dtype = float)

k = hkfields(zones_vec = zones_vec, 
              parameter_sets = parameter_sets,
              n_zones = n_zones
              )

# from gwf81x41 import rungwfmodle
# from gwf81x41 import plotgwm
# zones_vec81x41 = pd.read_csv('C:/Users/tian/Desktop/gwf_flopy_modflow6/input/original_zones_81x41.csv', dtype=np.float64)
# zones_vec = zones_vec81x41
# k = hkfields(zones_vec = zones_vec, 
#              parameter_sets = parameter_sets
#              )

k11 = np.exp(k)/100.0 * 86400
hk = k11
k33 = hk  # Vertical hydraulic conductivity ($m/d$)


# ax = plt.subplot()
# cmap = mpl.cm.viridis
# bounds = [1, 2, 3, 4, 5, 6]
# norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
# im = ax.imshow(hk,cmap=cmap, norm=norm)
# plt.title("Log(K) Field")
# plt.colorbar(im)
# plt.show()




rungwfmodle(hk)
plotgwm(hk)

waterhead = pd.read_csv('C:/Users/tian/Desktop/gwf_flopy_modflow6/model/gwf50x50/gwf-gwf50x50.obs.head.csv', dtype=np.float64)[-1:]
