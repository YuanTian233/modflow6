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
import pandas as pd
import numpy as np
import time

t0 = time.time()

model_output_path = 'C:/Users/tian/Desktop/gwf_mc' #remamber to change this path. this is where the code is.
exe_name = 'C:/Users/tian/Desktop/mf6.4.2/bin/mf6.exe'  #remamber to change this path.


if os.path.exists(model_output_path + '/output'):
    print('path exists')
else:
    os.mkdir(model_output_path + '/output')




from gwm import hkfields
from gwm import runmodelflow


n_zones = 6 
n_reali =  2


gm_mean = math.log(1e-5)  # ln(K) average --> constant
gm_var = 1.0  # ln(K) variance --> constant
parameter_sets = np.random.normal(loc=gm_mean, 
                                      scale=gm_var,
                                      size=(n_reali, n_zones)
                                      )

    
zones_vec50x50 = pd.read_csv((model_output_path + '/input/zone_distribution.csv'), dtype=np.float64)
zones_vec = np.array(zones_vec50x50, dtype = float)


df = pd.DataFrame(parameter_sets)        
df.to_csv(os.path.join(model_output_path+ r'/output','parameter_sets.csv'))



count = 0
model_name = "50"  

waterhead = []
concentration  = []
for count in range(n_reali):
    # tmpdir = tempfile.mkdtemp()
    
    sim_name = f'{model_name}_{count}' #simulation name of each model
    ws = os.path.join('model',sim_name) #worksapce for each model
    gwfname = "f" + sim_name # a name you can change for groundwater flow model. Name maximum length of 16
    gwtname = "t" + sim_name # a name you can change for groundwater transport model. Name maximum length of 16 
    
            
    k = hkfields(zones_vec = zones_vec,parameter_sets = parameter_sets[count:,],n_zones = n_zones)
        
    k11 = np.exp(k)/100.0 * 86400  # cm^2/s -> m^2/d
    hk = k11
    k33 = hk  # Vertical hydraulic conductivity ($m/d$)
    

    #call the modflow6    
    runmodelflow(sim_name = sim_name, 
                 sim_ws = ws,
                 exe_name = exe_name,
                 gwfname = gwfname,
                 gwtname = gwtname,
                 hk = hk,
                 # plot = False
                 )
 
    #read the waterhead and concentration for the last day(10000 day) for each model run.
    waterhead1 = pd.read_csv((os.getcwd() + f'/model/{sim_name}/{gwfname}.obs.head.csv'), dtype=np.float64, header = 0, usecols=range(1,11),skiprows=1)[-1:]
    waterhead1 = np.array(waterhead1)
    waterhead.append(waterhead1)
    
    concentration1 = pd.read_csv((os.getcwd() + f'/model/{sim_name}/{gwtname}.obs.conc.csv'), dtype=np.float64, header = 0, usecols=range(1,11),skiprows=1)[-1:]
    concentration1 = np.array(concentration1)
    concentration.append(concentration1)
    
    # shutil.rmtree('C:/Users/tian/Desktop/gp/testbal_mc/model/{sim_name}')
    count = count + 1 
    

   
#save the waterhead and concentration. 
waterhead=np.array(waterhead).reshape(n_reali,10)# in this case 10 obs locations has been seit, so here is reshape(n_reali,'10').
df = pd.DataFrame(waterhead)        
df.to_csv(os.path.join(model_output_path+ r'/output','waterhead.csv'))

concentration=np.array(concentration).reshape(n_reali,10)
df = pd.DataFrame(concentration)        
df.to_csv(os.path.join(model_output_path+ r'/output','concentration.csv'))


shutil.rmtree((os.getcwd() + '/model'))



print("--- %s seconds ---" % (time.time() - t0))


from gwm import add_noise
ret_error=True
# Measurement data:
h_error = [0.06, 0.01]
c_error = [0.06, 0.02]

# Extract results at observation locations

c_obs = pd.read_csv((os.getcwd() + r'/output/concentration.csv'), header = 0, index_col = 0)
h_obs = pd.read_csv((os.getcwd() + r'/output/waterhead.csv'),header = 0, index_col = 0)


h_obs = add_noise(h_obs, h_error)[0]
c_obs = add_noise(c_obs, c_error)[0]
    
