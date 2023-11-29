# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 14:57:27 2023


"""
import math
import pandas as pd
import numpy as np



'''
This is the main of the flopy_modflow6
the hk is a random value.
other parameters can be changed in gwf50x50
'''

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
from gwf50x50 import rungwfmodle
from gwf50x50 import plotgwm
# read the zone distribution
zones_vec50x50 = pd.read_csv('C:/Users/tian/Desktop/testzones/zone_distribution.csv', dtype=np.float64)
zones_vec = np.array(zones_vec50x50, dtype = float)

# assigns corresponding log(K) to each cells
k = hkfields(zones_vec = zones_vec, 
              parameter_sets = parameter_sets,
              n_zones = n_zones
              )


k11 = np.exp(k)/100.0 * 86400  # cm^2/s -> m^2/d
hk = k11
k33 = hk  # Vertical hydraulic conductivity ($m/d$)


rungwfmodle(hk)
plotgwm(hk)