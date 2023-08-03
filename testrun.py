# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 14:47:11 2023

@author: tian
"""
import os
import shutil
import matplotlib.pyplot as plt
import flopy
import math
import csv
import pandas as pd
import numpy as np
import scipy.stats as stats
from copy import deepcopy
import time

t0 = time.time()


from testmc import hkfields
from testmc import runmodelflow

n_zones = 6 
n_reali =  3


gm_mean = math.log(1e-5)  # ln(K) average --> constant
gm_var = 1.0  # ln(K) variance --> constant
parameter_sets = np.random.normal(loc=gm_mean, 
                                      scale=gm_var,
                                      size=(n_reali, n_zones)
                                      )

    
zones_vec50x50 = pd.read_csv('C:/Users/tian/Desktop/gwf/gwf50x50/input/zone_distribution.csv', dtype=np.float64)
zones_vec = np.array(zones_vec50x50, dtype = float)


df = pd.DataFrame(parameter_sets)        
df.to_csv('C:/Users/tian/Desktop/gp/testbal_mc/outputMC/parameter_sets.csv')



count = 0
model_name = "50"  #Name maximum length of 16
exe_name = 'C:/Users/tian/Desktop/mf6.4.2/bin/mf6.exe'  #remamber to change this path.
waterhead = []
concentration  = []
for count in range(n_reali):
    # tmpdir = tempfile.mkdtemp()
    
    name = f'{model_name}_{count}'
    ws = os.path.join('model',name)
    gwfname = "f" + name # a name you can change for groundwater flow model
    gwtname = "t" + name # a name you can change for groundwater transport model
    
    # ws = os.path.join(tmpdir,name)          
    k = hkfields(zones_vec = zones_vec,parameter_sets = parameter_sets[count:,],n_zones = n_zones)
        
    k11 = np.exp(k)/100.0 * 86400  # cm^2/s -> m^2/d
    hk = k11
    k33 = hk  # Vertical hydraulic conductivity ($m/d$)
    runmodelflow(sim_name = name, sim_ws = ws ,exe_name = exe_name ,gwfname=gwfname,gwtname=gwtname, hk = hk)
    
    waterhead1 = pd.read_csv(f'C:/Users/tian/Desktop/gp/testbal_mc/model/{name}/{gwfname}.obs.head.csv', dtype=np.float64, header = 0, usecols=range(1,11),skiprows=1)[-1:]
    waterhead1 = np.array(waterhead1)
    waterhead.append(waterhead1)
    
    concentration1 = pd.read_csv(f'C:/Users/tian/Desktop/gp/testbal_mc/model/{name}/{gwtname}.obs.conc.csv', dtype=np.float64, header = 0, usecols=range(1,11),skiprows=1)[-1:]
    concentration1 = np.array(concentration1)
    concentration.append(concentration1)
    count = count + 1 
    
   
waterhead=np.array(waterhead).reshape(n_reali,10)
df = pd.DataFrame(waterhead)        
df.to_csv('C:/Users/tian/Desktop/gp/testbal_mc/outputMC/waterhead.csv')

concentration=np.array(concentration).reshape(n_reali,10)
df = pd.DataFrame(concentration)        
df.to_csv('C:/Users/tian/Desktop/gp/testbal_mc/outputMC/concentration.csv')


# shutil.rmtree('C:/Users/tian/Desktop/gp/testbal_mc/model')



print("--- %s seconds ---" % (time.time() - t0))
    

     

# waterhead=np.array(waterhead).reshape(n_reali,10)
# df = pd.DataFrame(waterhead)        
# # df.to_csv('C:/Users/tian/Desktop/gp/testbal_mc/outputMC/waterhead.csv')

# concentration=np.array(concentration).reshape(n_reali,10)
# df = pd.DataFrame(concentration)        
# # df.to_csv('C:/Users/tian/Desktop/gp/testbal_mc/outputMC/concentration.csv')


# shutil.rmtree('C:/Users/tian/Desktop/gp/testbal_mc/model')