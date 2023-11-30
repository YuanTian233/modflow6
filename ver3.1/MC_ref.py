# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 14:47:11 2023
This is just MC, it can run this model for many times to get ref data

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
import scipy.stats as stats
import scipy.io
t0 = time.time()

outputfolder = 'output'
model_output_path = os.getcwd()
if os.path.exists(os.path.join(model_output_path, outputfolder)):
    print('Path exists')
else:
    os.mkdir(os.path.join(model_output_path, outputfolder))



from gwm import hkfields
from gwm import runmodelflow


 
n_reali =  1 #this is the MC size


n_zones = 6 # i got a zoned model, so i have 6 parameter

gm_mean = math.log(1e-5)  # ln(K) average --> constant
gm_var = 1.0  # ln(K) variance --> constant
parameter_sets = np.random.normal(loc=gm_mean, 
                                      scale=gm_var,
                                      size=(n_reali, n_zones)
                                      )
obs = pd.read_csv((model_output_path + '/input/obs_hk.csv'),header = 0, index_col = 0)
obs= np.array(obs)
    
zones_vec50x50 = pd.read_csv((model_output_path + '/input/zone_distribution.csv'), dtype=np.float64)
zones_vec = np.array(zones_vec50x50, dtype = float)

# save the parameter sets
df = pd.DataFrame(parameter_sets)        
df.to_csv(os.path.join(model_output_path+ r'/output','parameter_sets.csv'))

####if you want the ture results you can use this data as the input in line 77
mat_file_path = (model_output_path + '/ref_data/Y_true.mat')
# Load the .mat file
mat_contents = scipy.io.loadmat(mat_file_path)
y_true = mat_contents['Y_true']

count = 0
waterhead = []
concentration  = []

for count in range(n_reali):
    # tmpdir = tempfile.mkdtemp()
    model_name = "50"  
    exe_name = 'C:/Users/tian/Desktop/mf6.4.2/bin/mf6.exe'  #remamber to change this path.
    sim_name = f'{model_name}_{count}' #simulation name of each model
    ws = os.path.join('model',sim_name) #worksapce for each model
    gwfname = "f" + sim_name # a name you can change for groundwater flow model. Name maximum length of 16
    gwtname = "t" + sim_name # a name you can change for groundwater transport model. Name maximum length of 16 
    
            
    k = hkfields(zones_vec = zones_vec,parameter_sets = obs,n_zones = n_zones)
        
    k11 = np.exp(k)/100.0 * 86400  # cm^2/s -> m^2/d
    # k11 = np.exp(y_true)/100.0 * 86400  # cm^2/s -> m^2/d
    hk = k11
    k33 = hk  # Vertical hydraulic conductivity ($m/d$)
    

    #call the modflow6    
    runmodelflow(sim_name = sim_name, 
                 sim_ws = ws,
                 exe_name = exe_name,
                 gwfname = gwfname,
                 gwtname = gwtname,
                 hk = hk,
                 plot = True
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


# shutil.rmtree('{modellpath}/model')



print("--- %s seconds ---" % (time.time() - t0))



    
