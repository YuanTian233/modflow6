
for groundwater flow and transport model

Markov Chain Monte Carlo (MCMC)

Monte Carlo (MC)

need flopy and modflow6

MODFLOW 6 is used in this model. 
        
MODFLOW 6:
        https://water.usgs.gov/water-resources/software/MODFLOW-6/
        
More example questions please check this website:
        https://modflow6-examples.readthedocs.io/en/latest/notebook_examples.html






############################################################

floder:

##

input: obs data and zone distubution 

ref_data: 10,000 MODFLOW 6 runs and ture HK genareted by MATLAB

##############################



py file:

##

gwm.py: MODFOW 6 code, 50 by 50 cells, Hydraulic conductivity is the input

###############

bayesian_inference.py: all functions that the code need

###############

MC_ref.py: it can run MODFLOW 6 many times

###############

BAL and GPEs:

        expolartion sets sampling:
        
                bal_prior_MCMC.py: MCMC sampling
                
                bal_prior_mc.py: random sampling
                
        ##
        
        expolartion sets sampling:
        
                bal_post_mc.py: MC and rejection sampling
                
                bal_post_mcmc.py: MCMC sampling

###############

visualiztion:

plots.py

vs.py


############################################################




All data saved here:
https://zenodo.org/records/10220360?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjhmMTY5MjQwLTM3YzgtNDUzYy05NzVkLWNhYWJmNDNlZmZhMCIsImRhdGEiOnt9LCJyYW5kb20iOiJjYTM3ZTE5ZjMyMjgwMTZhYzAyN2Q0MGEzOTFhMTAwYiJ9.Q57E7ceCq_zRnz21ummDeKm4wqdbbck6NN6K4wSPwQLylHELRcI2-r9Kjua5G_OS02Mve08uzzuNnB39KBpszg
