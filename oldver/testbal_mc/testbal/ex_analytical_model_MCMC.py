"""
Module trains a GPR with BAL for a non-linear analytical model.

Reference for non-linear model:
Oladyshkin S, Nowak W. The Connection between Bayesian Inference and Information Theory for Model Selection,
Information Gain and Experimental Design. Entropy. 2019; 21(11):1081. https://doi.org/10.3390/e21111081
"""
import numpy as np
from scipy.stats import uniform
import scipy.stats as stats
# import chaospy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from bayesian_inference import calculate_likelihood, calculate_likelihood_manual, compute_bme, bal_selection_criteria
from plots import plot_size, plot_1d_gpe_final, plot_gpe_scores, plot_bal_criteria



def nonlinear_model(t, params):
    """
    Multi-dimensional non-linear model.
    Reference: ladyshkin S, Nowak W. The Connection between Bayesian Inference and Information Theory for Model
    Selection, Information Gain and Experimental Design. Entropy. 2019; 21(11):1081. https://doi.org/10.3390/e21111081

    :param t: array [n_obs, ]
        with 't' values where the function is to be evaluated
    :param params: array [mc_samples, ndim]
        with 'mc_samples' number of parameter sets to evaluate in the analytical function
    :return: array [mc_samples, n_obs]
        With model outputs, evaluated in each observation location and for each parameter set.

    Note: Function needs at least 2 parameters. If ndim == 1, the same parameter is used as parameters 1 and 2.
    """

    # If vector with parameters is passed, reshape it to matrix
    if params.ndim == 1:
        params = np.reshape(params, (1, params.shape[0]))

    # If only one parameter is sent, copy it to same the same one as parameter 1 and parameter 2
    if params.shape[1] == 1:
        params = np.hstack((params, params))

    ndim = params.shape[1]  # number of parameters
    n_sets = params.shape[0]  # number of parameter sets
    nobs = t.shape[0]         # number of observations

    term1 = (params[:, 0] ** 2 + params[:, 1] - 1) ** 2
    term2 = params[:, 0] ** 2
    term3 = 0.1 * params[:, 0] * np.exp(params[:, 1])

    # Term that all models have in common:
    term5 = 0
    if ndim > 2:
        for i in range(2, ndim):
            term5 = term5 + np.power(params[:, i], 3) / (i + 1)

    # Sum all non-time-related terms: gives one value per row, and one row for each parameter set
    const_per_set = term1 + term2 + term3 + term5 + 1  # All non-time-related terms

    # Calculate time term: gives one value per row for each time interval
    term4 = np.full((n_sets, nobs), 0.0)
    for i in range(0, n_sets):
        term4[i, :] = -2 * params[i, 0] * np.sqrt(0.5 * t)

    output = term4 + np.repeat(const_per_set[:, None], nobs, axis=1)

    return output


# ------------------------------------------------------------------------------------------------------------------ #
# --------------------------------------- USER INPUT --------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #

# Analytical model data:
n_dim = 3 # Number of uncertain parameter sets. (2, 10)
distribution = 'uniform'    # Prior distribution for all uncertain parameters
dist_parameters = [-5, 5]   # range of values for uniform distribution

# Observations (in a real case this would be read directly after computing the full-complexity model)
t_ = np.arange(0, 1.01, 1 / 9)  # for my analytical model
# t = np.array([[0.01]])
n_obs = len(t_)            # number of points (space or time) I am going to use in my calibration
measurement_error = np.full(n_obs, 2)   # measurement error associated to each observation (constant for all here)

synthetic_solution = np.full((1, n_dim), 0.0)           # synthetic true parameter values
observations = nonlinear_model(t_, synthetic_solution)   # synthetic true observations

# Bayesian inference data:
mc_size = 10_000    # number of samples to get from the prior distribution

# Bayesian Active Learning data:
tp_i = 5                 # number of initial collocation points
iteration_limit = 5      # number of BAL iterations
d_size_AL = 1_000        # number of active learning sets (sets I take from the prior to do the active learning).
mc_size_AL = 1_000      # sample size for output space
# Note: d_size_AL+ iteration_limit < mc_size
al_strategy = "RE"   # BAL criteria (Here we want to test different methods)

# Gaussian Process data:
alpha = 0.0002                # Noise added to diagonal in GPE training
n_restarts = 10               # Number of optimizer starts
y_norm_gpe = True             # normalization for GPE training


# ------------------------------------------------------------------------------------------------------------------ #
# --------------------------------------- Sample from prior -------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #

# prior
if distribution == 'uniform':
    # Get a large sample from the prior
    # prior = np.random.uniform(dist_parameters[0], dist_parameters[1], size=(mc_size, n_dim))
    prior = uniform.rvs(size=(mc_size, n_dim), loc=dist_parameters[0], scale=dist_parameters[1]-dist_parameters[0])

    # Get prior probability for each independent parameter value in each set
    prior_pdf_ind = uniform.pdf(prior, loc=dist_parameters[0], scale=dist_parameters[1]-dist_parameters[0])
    # Get prior probability for each set
    prior_pdf_set = np.prod(prior_pdf_ind, axis=1)

    # Get collocation points: (here we use a random sampling, we can think of using sobol sampling with chaospy
    collocation_points = uniform.rvs(size=(tp_i, n_dim), loc=dist_parameters[0],
                                           scale=dist_parameters[1] - dist_parameters[0])

# Evaluate collocation points to get output of training set
model_evaluation = nonlinear_model(t=t_, params=collocation_points)


# ------------------------------------------------------------------------------------------------------------------ #
# --------------------------------------- Get reference data-------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #
# Data to compare the surrogate model to
ref_output = nonlinear_model(t=t_, params=prior)
cov_mat = np.diag(measurement_error**2)
mean = observations
a = len(mean)
likelihood = stats.multivariate_normal.pdf(ref_output, cov=cov_mat, mean=observations[0, :])
ref_bme, ref_re, ref_ie = compute_bme(model_predictions=ref_output, observations=observations, var=measurement_error**2,
                                      prior_pdf=prior_pdf_set, prior_samples=prior)

# ------------------------------------------------------------------------------------------------------------------ #
# --------------------------------------- Train GPR -------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------------------ #


# Part 1. Initialization of information  ------------------------------------------------------------------------------
BME = np.zeros((iteration_limit+1, 1))
RE = np.zeros((iteration_limit+1, 1))
IE = np.zeros((iteration_limit+1, 1))
al_BME = np.zeros((d_size_AL, 1))
al_RE = np.zeros((d_size_AL, 1))
al_IE = np.zeros((d_size_AL, 1))
graph_list = []
graph_name = []
crit_val = np.zeros((iteration_limit, 1))

# Loop for Bayesian update
for Ntp in range(0, iteration_limit+1):
    surrogate_prediction = np.zeros((n_obs, mc_size))
    surrogate_std = np.zeros((n_obs, mc_size))
    gp_list = []

    # Loop for each observation, which gets its own surrogate model
    for i, model in enumerate(model_evaluation.T):
        # 1. Set up kernel
        kernel = np.var(model) * RBF(length_scale=1.0, length_scale_bounds=(1e-25, 1e15))
        a = np.var(model)
        # setup gp
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0002, normalize_y=True, n_restarts_optimizer=10)
        # Train GP
        gp.fit(collocation_points, model)

        # Evaluate all prior parameter sets in GP
        surrogate_prediction[i, :], surrogate_std[i, :] = gp.predict(prior, return_std=True)

        # Save trained GP to use later
        gp_list.append(gp)
        # =============================================================================
        #         print(f"    Looping through loc {i}/{n_obs} --> hp: {gp.kernel_.theta}")
        # =============================================================================

    # Computation of bayesian scores (in parameter space) -----------------------------------------------------
    total_error = (measurement_error ** 2) * np.ones(observations.shape[0])
    BME[Ntp], RE[Ntp], IE[Ntp] = compute_bme(surrogate_prediction.T, observations, total_error,
                                             prior_pdf=prior_pdf_set, prior_samples=prior)

    # Bayesian active learning (in output space) --------------------------------------------------------------
    if Ntp < iteration_limit:
        # Index of the elements of the prior distribution that have not been used as collocation points
        aux1 = np.where((prior[:d_size_AL + Ntp, :] == collocation_points[:, None]).all(-1))[1]
        aux2 = np.invert(np.in1d(np.arange(prior[:d_size_AL + Ntp, :].shape[0]), aux1))
        al_unique_index = np.arange(prior[:d_size_AL + Ntp, :].shape[0])[aux2]

        for iAL in range(0, len(al_unique_index)):
            # Exploration of output subspace associated with a defined prior combination.
            al_exploration = np.random.normal(size=(mc_size_AL, n_obs)) * surrogate_std[:, al_unique_index[iAL]] + \
                             surrogate_prediction[:, al_unique_index[iAL]]

            # BAL scores computation
            al_BME[iAL], al_RE[iAL] = compute_bme(model_predictions=al_exploration, observations=observations,
                                                  var=total_error)

        # Part 7. Selection criteria for next collocation point ------------------------------------------------------
        al_value, al_value_index = bal_selection_criteria(al_strategy, al_BME, al_RE)
        crit_val[Ntp, 0] = al_value

        # Part 8. Selection of new collocation point
        collocation_points = np.vstack((collocation_points, prior[al_unique_index[al_value_index], :]))

        # Part 9. Computation of the numerical model in the newly defined collocation point --------------------------
        model_evaluation = np.vstack((model_evaluation, nonlinear_model(t=t_, params=collocation_points[-1, :])))

    # Progress report
    print("Bayesian iteration: " + str(Ntp + 1) + "/" + str(iteration_limit))




import pzmc as pm

# Plot results:
# Get confidence intervals
lower_ci = surrogate_prediction - 2*surrogate_std
upper_ci = surrogate_prediction + 2*surrogate_std

if n_dim == 1:
    plot_1d_gpe_final(prior, surrogate_prediction,  lower_ci, upper_ci, collocation_points,
                      model_evaluation, tp_i, iteration_limit, observations)

plot_gpe_scores(BME, RE, tp_i, ref_bme=ref_bme, ref_re=ref_re)
plot_bal_criteria(crit_val, al_strategy)

stop = 1

