# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 15:58:49 2023

@author: tian
"""
import numpy as np
import pandas as pd
import math
import csv

n_reali =  2
n_zones = 6 
alist = []
for count in range(n_reali):
    gm_mean = math.log(1e-5)  # ln(K) average --> constant
    gm_var = 1.0  # ln(K) variance --> constant
    parameter_sets = np.random.normal(loc=1.0, 
                                          scale=gm_var,
                                          size=(1, n_zones)
                                          )
    alist.append(parameter_sets)
    
    
    # print(alist)


df=pd.DataFrame(alist[0])
for i in range(len(alist)-1):
    df1=pd.DataFrame(alist[i+1])
    df=pd.concat([df,df1],axis=0)

# a = np.reshape(alist,(n_reali,6))
# df=pd.DataFrame(a)
# # with open("output.txt", "w") as f:
# #     writer = csv.writer(f, delimiter=' ')
# #     writer.writerows(alist)
# # np.savetxt('output.csv', alist, delimiter=",")
# # df.to_csv("data",index=False)