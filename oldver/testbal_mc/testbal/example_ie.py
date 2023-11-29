import numpy as np
import scipy.stats as stats
from gpr_bal.bayesian_inference import *

"""
I assume I have 10 uncertain parameters, which all have a standard gaussian distribution N(0, 1), ans I sample 100 
samples. 
I usie scipy.stats to sample. 
"""
# Here I will not estimate the model_y, observations or error. You use the ones from your example.
model_y = []  # model outputs, one for each output location/type
obs = []      # observations, one for each output location/type
error = []     # observation errors, one for each obs

# Option 1: Sample parameter sets: using scipy.stats
# prior = stats.norm.rvs(loc=0, scale=1, size=(10, 100))
# prior_prob = stats.norm.pdf(x=prior, loc=0, scale=1)

# option 2: I create a stats.norm object, and then I call it to get samples (rvs) and get probability density function
# (pdf)
dist = stats.norm(loc=0, scale=1)
prior = dist.rvs(size=(10, 100))
prior_prob = dist.pdf(x=prior)

# Since the prior_pdf/prior_prob provides one for each parameter, and we need one for each parameter set, we get the
# joint probability: since they are independent, we multiply the probabilities.
prior_prob_per_sample = np.mean(prior_prob, axis=0)  #axis=0, we want to average the rows, and get 1 for each sample

# The IE equation needs the log(post_pdf), so we transform it to log_pdf_
log_prior_prob = np.log(prior_prob_per_sample)

bme, re, ie = compute_bme(model_predictions=model_y, observations=obs, var=error,
                          prior_pdf=log_prior_prob, prior_samples=prior,      # These are needed for estimating IE
                          posterior_sampling='rejection_sampling',
                          compute_likelihood='auto')

stop = 1


