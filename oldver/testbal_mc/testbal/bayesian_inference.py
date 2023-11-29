import numpy as np
import math
import scipy.stats as stats


def calculate_likelihood(self):
    """
    Function calculates likelihood between measured data and the model output using the scipy.stats module equations.

    Notes:
    * Generates likelihood array with size [MCx1].
    * Likelihood function is multivariate normal distribution, considering independent and Gaussian-distributed
    errors.
    """

    self.likelihood = stats.multivariate_normal.pdf(self.model_predictions, cov=self.cov_mat,
                                                    mean=self.observations[0, :])


def calculate_likelihood_manual(model_predictions, observations, cov_mat):
    """
    Function calculates likelihood between observations and the model output manually, using numpy calculations.

    Notes:
    * Generates likelihood array with size [MCxN], where N is the number of measurement data sets.
    * Likelihood function is multivariate normal distribution, considering independent and Gaussian-distributed
    errors.
    * Method is faster than using stats module ('calculate_likelihood' function).
    """
    # Calculate constants:
    det_R = np.linalg.det(cov_mat)
    invR = np.linalg.inv(cov_mat)
    const_mvn = pow(2 * math.pi, - observations.shape[1] / 2) * (1 / math.sqrt(det_R))  # ###########

    # vectorize means:
    means_vect = observations[:, np.newaxis]  # ############

    # Calculate differences and convert to 4D array (and its transpose):
    diff = means_vect - model_predictions  # Shape: # means
    diff_4d = diff[:, :, np.newaxis]
    transpose_diff_4d = diff_4d.transpose(0, 1, 3, 2)

    # Calculate values inside the exponent
    inside_1 = np.einsum("abcd, dd->abcd", diff_4d, invR)
    inside_2 = np.einsum("abcd, abdc->abc", inside_1, transpose_diff_4d)
    total_inside_exponent = inside_2.transpose(2, 1, 0)
    total_inside_exponent = np.reshape(total_inside_exponent,
                                       (total_inside_exponent.shape[1], total_inside_exponent.shape[2]))

    likelihood = const_mvn * np.exp(-0.5 * total_inside_exponent)

    # Convert likelihoods to vector:
    if likelihood.shape[1] == 1:
        likelihood = likelihood[:, 0]

    return likelihood


def compute_bme(model_predictions, observations, var, prior_pdf=None, prior_samples=None,
                posterior_sampling='rejection_sampling',
                compute_likelihood='auto'):
    """
    Does Bayesian inference and estimates BME, ELPD and RE, based on the input data
    :param model_predictions: array [mc_size, nobs]
        With output values from model
    :param observations: array [1, nobs]
        with true observation values
    :param var: array[nobs,]
        with variance (error**2) associated to each observation
    :param prior_pdf: array[mc_size, ]
        with prior probability associated to each prior parameter set
    :param prior_samples: array [mc_size, ndim]
        With prior samples used to estimate 'model_predictions'. Not necessary
    :param posterior_sampling: string
        With method to use to sample from the posterior. Default is 'rejection sampling'
    :param compute_likelihood: string
        Method to estimate the likelihood. Default is 'auto' to use scipy.stats. 'manual' can be used for speed up.
    :return: float, float
        BME and RE values
    """
    # Calculate covariance
    cov_mat = np.diag(var)

    # 1. Calculate Gaussian likelihood ............................................................................
    if 'auto' in compute_likelihood.lower():
        likelihood = stats.multivariate_normal.pdf(model_predictions, cov=cov_mat, mean=observations[0, :])
    else:
        likelihood = calculate_likelihood_manual(model_predictions, observations, cov_mat)

    # 2. Rejection sampling --------------------------------------------------------------------------------------- #
    if posterior_sampling == 'rejection_sampling':
        random_variables = stats.uniform.rvs(size=model_predictions.shape[0])  # random numbers
        max_likelihood = np.max(likelihood)  # Max likelihood

        # Eq. 2 (Oladyshkin and Nowak)
        BME = np.mean(likelihood)

    if max_likelihood > 0:
        # 1. Get indexes of likelihood values whose normalized values < RN
        posterior_index = np.array(np.where(likelihood / max_likelihood > random_variables)[0])
        # 2. Get posterior_likelihood:
        posterior_likelihood = np.take(likelihood, posterior_index, axis=0)

        # 3. Get posterior values ..................................................................................
        if posterior_likelihood.shape[0] > 0:
            # Output sample of the model outputs
            posterior_output = np.take(model_predictions, posterior_index, axis=0)
            if prior_samples is not None:
                # Posterior parameter samples
                posterior_samples = np.take(prior_samples, posterior_index, axis=0)

        # Calculate posterior criteria # Eq. (5) - Oladyshkin and Nowak
        ELPD = np.mean(np.log(posterior_likelihood))

        # Calculate relative entropy: Eqs. (7), (13) --> (42)
        RE = ELPD - np.log(BME)

        if prior_pdf is not None:
            # CE and IE can only be estimated if a prior_pdf is given
            posterior_pdf = np.take(prior_pdf, posterior_index, axis=0)
            # Calculate cross entropy, to use in IE: Eq. (4), 2nd term in eq. (38)
            CE = np.mean(np.log(posterior_pdf))
            # Calculate information entropy: Eqs. (3) --> (38)
            IE = -RE - CE
        else:
            IE = None

    else:
        # If all likelihoods are 0, then log(BME) cannot be calculated, and will give an error, therefore we assign
        # a value of 0 to all scores.
        ELPD = 0
        RE = 0
        IE = 0

    if prior_pdf is None:
        return BME, RE
    else:
        return BME, RE, IE


def bal_selection_criteria(al_strategy, al_BME, al_RE, al_IE):
    """
    Gives the best value of the selected bayesian score and index of the associated parameter combination
    ----------
    al_strategy : string
        strategy for active learning, selected bayesian score. Options: 'BME', 'RE' , 'IE', 'postBME', 'postRE' and 'postIE'
    al_BME : array [d_size_AL,1]
        bayesian model evidence of active learning sets
    al_RE: array [d_size_AL,1]
        relative entropy of active learning setsc 

    Returns
    -------
    al_value: float
        best value of the selected bayesian score
    al_value_index: int
        index of the associated parameter combination

    Notes:
    * d_size_AL is the number of active learning sets (sets I take from the prior to do the active learning)

    ToDO: Add IE as BAL criteria here.
    """


    if al_strategy == "BME":
        # al_value = np.amax(al_BME)
        # al_value_index = np.argmax(al_BME)
        al_value = np.nanmax(al_BME)
        al_value_index = np.nanargmax(al_BME)

        if np.amax(al_BME) == 0:
            print("Warning Active Learning: all values of Bayesian model evidences equal to 0")
            print("Active Learning Action: training point have been selected randomly")

    elif al_strategy == "RE":

        # al_value = np.amax(al_RE)
        # al_value_index = np.argmax(al_RE)
        al_value = np.nanmax(al_RE)
        al_value_index = np.nanargmax(al_RE)
        
        if np.nanmax(al_RE) == 0 and np.nanmax(al_BME) != 0:
            al_value = np.nanmax(al_BME)
            al_value_index = np.nanargmax(al_BME)
            print("Warning Active Learning: all values of Relative entropies equal to 0")
            print("Active Learning Action: training point have been selected according Bayesian model evidences")
        elif np.nanmax(al_RE) == 0 and np.nanmax(al_BME) == 0:
            al_value = np.nanmax(al_BME)
            al_value_index = np.nanargmax(al_BME)
            print("Warning Active Learning: all values of Relative entropies equal to 0")
            print("Warning Active Learning: all values of Bayesian model evidences are also equal to 0")
            print("Active Learning Action: training point have been selected randomly")

        # if np.amax(al_RE) == 0 and np.amax(al_BME) != 0:
        #     al_value = np.amax(al_BME)
        #     al_value_index = np.argmax(al_BME)
        #     print("Warning Active Learning: all values of Relative entropies equal to 0")
        #     print("Active Learning Action: training point have been selected according Bayesian model evidences")
        # elif np.amax(al_RE) == 0 and np.amax(al_BME) == 0:
        #     al_value = np.amax(al_BME)
        #     al_value_index = np.argmax(al_BME)
        #     print("Warning Active Learning: all values of Relative entropies equal to 0")
        #     print("Warning Active Learning: all values of Bayesian model evidences are also equal to 0")
        #     print("Active Learning Action: training point have been selected randomly")
        
    elif al_strategy == 'IE' or al_strategy == 'postIE':
        # Find the indices of the non-NaN values
        non_nan_indices = np.where(~np.isnan(al_IE))[0]

        # Find the index of the non-NaN value closest to 0
        al_value_index = non_nan_indices[np.argmin(np.abs(al_IE[non_nan_indices] - 0))]

        # Get the non-NaN value closest to 0
        al_value = al_IE[al_value_index]
        # al_value = np.min(al_IE)
        # al_value_index = np.argmin(al_IE)
    # if al_strategy == 'postIE':
    #     al_value = np.nanmin(al_IE)
    #     al_value_index = np.nanargmin(al_IE)
    
    elif al_strategy == 'postRE':
        # Filter non-NaN and non-Inf values
        valid_values = al_RE[np.isfinite(al_RE)]

        # Get the index of the biggest non-NaN and non-Inf value in the original array
        al_value_index = np.where(al_RE == valid_values[np.argmax(valid_values)])[0][0]

        # Get the biggest non-NaN and non-Inf value
        al_value = valid_values[np.argmax(valid_values)]

    elif al_strategy == 'postBME':
        # Filter non-NaN and non-Inf values
        valid_values = al_BME[np.isfinite(al_BME)]
    
        # Get the index of the biggest non-NaN and non-Inf value in the original array
        al_value_index = np.where(al_BME == valid_values[np.argmax(valid_values)])[0][0]
    
        # Get the biggest non-NaN and non-Inf value
        al_value = valid_values[np.argmax(valid_values)] 
        
       
       
    
    return al_value, al_value_index


def compute_post_bme(
                     model_predictions, 
                     observations, 
                     var,
                     covariance_matrices,
                     post_pdf,

                     # prior_samples=None,
                     # likelihood,
                     # posterior_sampling='MCMC_sampling',
                     
                     ):
    """
    Does Bayesian inference and estimates post-BME, post-RE and post-iE, based on the input data
    n is the number of uncertain parameters.
    :param model_predictions: array [mc_size, nobs]
        With output values from model
    :param observations: array [1, nobs]
        with true observation values
    :param var: array[nobs,]
        with variance (error**2) associated to each observation
    :param post_pdf: array[mc_size, ]
        with posterior probability associated to each prior parameter set
    :param prior_samples: array [mc_size, ndim]
        With prior samples used to estimate 'model_predictions'. Not necessary
    :param covariance_matrices: [n, n]
        from Equation (27) 'doi.org/10.3390/e21111081'
    :return: float, float
        postBME, postRE, and psotIE values
    """
    postIE = 0.5 * (
        np.log(
        (
            (
                2 * math.pi * math.exp(1)
                )**6
            ) 
        * np.linalg.det(covariance_matrices)
        )
        )

    
    # Calculate 
    

    
    cov_mat = np.diag(var)
    # 1. Calculate Gaussian likelihood ............................................................................
    likelihood = stats.multivariate_normal.pdf(model_predictions, cov=cov_mat, mean=observations[0, :])
    # if posterior_sampling == 'MCMC_sampling':
    # 
    #     likelihood = likelihood
    # else:
        # likelihood = stats.multivariate_normal.pdf(model_predictions, cov=cov_mat, mean=observations[0, :])
    postBME = np.exp(np.mean(np.log(likelihood)) + np.mean(np.log(post_pdf)) + postIE)       
    postRE = -np.mean(np.log(post_pdf))- postIE



        
        
    return postIE, postBME, postRE
    
def MCMC_MH_post(al_old,
                 mc_size_AL,
                 # model_predictions, 
                 observations, 
                 variance,
                 mean,
                 total_error,

                        # prior_samples=None,
                        # likelihood,
                        # posterior_sampling='MCMC_sampling',
                        
                        ): 
    '''
    
    Does MCMC sampling.
    :param al_old: array[nobs, 1]
        starting piont
    :param mc_size_AL 
        MC size
    :param observations: array[nobs,1]
        with variance (error**2) associated to each observation
    :param variance:  
         
    :param mean
    
    :param total_error: array[mc_size, ]
        error
      
    :return: float
        al_exploration
    
    '''
    al_exploration = np.zeros((mc_size_AL, 20))
    llk = np.zeros((mc_size_AL, 1))
    al_old =al_old
    for MC in range(mc_size_AL):
        al_new= stats.multivariate_normal(mean = al_old, 
                                              cov=0.05).rvs() 
                                            # 0.005 0.05 0.0005 0.00009 0.00007 0.000009
    
    
    
        

        old_pdf_set = stats.multivariate_normal.logpdf(al_old,mean=mean,cov = variance)
        new_pdf_set = stats.multivariate_normal.logpdf(al_new,mean=mean,cov = variance)
        
        llk_old = np.sum(stats.multivariate_normal.logpdf(al_old,  
                                                  cov=np.diag(total_error),
                                                  mean=observations[0, :]))
        llk_new = np.sum(stats.multivariate_normal.logpdf(al_new, 
                                                  cov= np.diag(total_error), 
                                                  mean= observations[0, :]))
        
        
        
        

        α = np.exp(
            np.log(llk_new) + new_pdf_set - np.log(llk_old) - old_pdf_set
            ) #+ prop_ratio)
        
        if np.random.random() < min(1,α):
            al_exploration[MC]  = al_new
            al_old = al_new
            llk[MC] = llk_new

    
    
        else:
            al_exploration[MC]  = al_old
            al_old = al_old 
            llk[MC] = llk_old

    return al_exploration
