# The purpose of this script is to use a Bayes Risk Map from Kalman Filtering, 
# and to use the same choices of random hyperparameters and truths to create
# the equivalent optimisation problem for an autoregressive Kalman Filter with 
# LS Filter weights.

import sys
import os
import numpy as np
import GPy

from plotting_tools.case_data_explorer import CaseExplorer as cs

test_case = int(sys.argv[1])
variation = int(sys.argv[2])
SigmaMx = float(sys.argv[3])
RMx  = float(sys.argv[4]) 
filepath = sys.argv[5]

######
filepath_LS_Ensemble =  filepath+'/LS_Ensemble_Folder/test_case_'+str(test_case)+'_var_'+str(variation)

# We now load truths from BR Maps
kf_original_obj = cs(test_case, variation, 1, 50, filepath, Hard_load='Yes', AKF_load='No')

####
# Initial HyperParameter Values 
####
period_0 = float(kf_original_obj.n_train)
length_scale_0 = kf_original_obj.Delta_T_Sampling*3.0
R_0 = 1.0
sigma_0 = 1.0
GPR_opt_params = []

####
# Initialise Kernel
####



idx_randparams = 0 # OR RANDOMLY SELECT FROM 0 - 74

prediction_errors = [] 
forecastng_errors = [] 

for idx_d in xrange(kf_original_obj.max_it_BR):

    # Load truth, make msmts
    init_ = kf_original_obj.random_hyperparams_list[idx_randparams, :]
    truth = kf_original_obj.macro_truth[idx_randparams, idx_d, :]
    y_signal = truth + kf_original_obj.msmt_noise_variance*np.random.randn(truth.shape[0])
    
    # Create training data objects and test pts for GPy
    X = kf_original_obj.Time_Axis[0:kf_original_obj.n_train, np.newaxis]
    Y = y_signal[0:kf_original_obj.n_train, np.newaxis]
    testx = kf_original_obj.Time_Axis[kf_original_obj.n_train - kf_original_obj.n_testbefore : ]
    
    # Reset GPy Model
    kernel_per = GPy.kern.StdPeriodic(1, period=period_0, variance=sigma_0, lengthscale=length_scale_0)
    gauss = GPy.likelihoods.Gaussian(variance=R_0)
    exact = GPy.inference.latent_function_inference.ExactGaussianInference()
    m1 = GPy.core.GP(X=X, Y=Y, kernel=kernel_per, likelihood=gauss, inference_method=exact)
    m1.std_periodic.variance.constrain_bounded(0, SigmaMx)
    m1.Gaussian_noise.variance.constrain_bounded(0, RMx)
    
    # Optimise GPy Model
    print('Run: ', idx_d)
    print('Before Optimisation: ', [m1.std_periodic.variance[0], m1.Gaussian_noise.variance[0], m1.std_periodic.period[0], m1.std_periodic.lengthscale[0]])
    m1.optimize([m1.std_periodic.variance[0], m1.Gaussian_noise.variance[0], m1.std_periodic.period[0], m1.std_periodic.lengthscale[0]])
    GPR_opt_params.append([m1.std_periodic.variance[0], m1.Gaussian_noise.variance[0], m1.std_periodic.period[0], m1.std_periodic.lengthscale[0]])
    print('After Optimisation: ', [m1.std_periodic.variance[0], m1.Gaussian_noise.variance[0], m1.std_periodic.period[0], m1.std_periodic.lengthscale[0]])
    
    # Make predictions using Optimised GPy  
    gpr_predictions = m1.predict(testx[:,np.newaxis])[0].flatten()
    
    # Calc RMS error
    truth_ = truth[kf_original_obj.n_train - kf_original_obj.n_testbefore : kf_original_obj.n_train + kf_original_obj.n_predict]
    residuals_sqr_errors = (gpr_predictions - truth_)**2

    prediction_errors.append(residuals_sqr_errors[0: kf_original_obj.n_testbefore])
    forecastng_errors.append(residuals_sqr_errors[kf_original_obj.n_testbefore : ])

np.savez(filepath_LS_Ensemble+'_GPR_PER_',
                msmt_noise_variance= kf_original_obj.msmt_noise_variance, 
                max_it_BR = kf_original_obj.max_it_BR, 
                macro_truth=kf_original_obj.macro_truth[idx_randparams, :, :],
                GPR_opt_params=GPR_opt_params,
                GPR_PER_prediction_errors=prediction_errors, 
                GPR_PER_forecastng_errors=forecastng_errors)
