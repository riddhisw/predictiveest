# The purpose of this script is to use a Bayes Risk Map from Kalman Filtering, 
# and to use the same choices of random hyperparameters and truths to create
# the equivalent optimisation problem for an autoregressive Kalman Filter with 
# LS Filter weights.

import sys
import os
import numpy as np

from kf.armakf import autokf as akf
from kf.fast_2 import kf_2017 as skf
from analysis_tools.case_data_explorer import CaseExplorer as cs

test_case = int(sys.argv[1])
variation = int(sys.argv[2])
filepath = sys.argv[3]
filepath_LS_Ensemble =  filepath+'/LS_Ensemble_Folder/test_case_'+str(test_case)+'_var_'+str(variation)

# We now load truths, hyperams, and noisy data from BR Maps

kf_original_obj = cs(test_case, variation, 1, 50, filepath, Hard_load='Yes')
descriptor = os.path.join(filepath, kf_original_obj.filename0+'_AKF_')

choose_one = 0 # or randomise, or take a mean across ensemble
stp_fwd = 1 # weights correspond to one step ahead predictions to match AR(q) formalism
skip_msmts = 1

weights = np.load(filepath_LS_Ensemble+'_LS_Ensemble.npz')['macro_weights'][choose_one, stp_fwd,:, 0]

macro_prediction_errors = [] 
macro_forecastng_errors = [] 

for idx_randparams in xrange(kf_original_obj.num_randparams):

    prediction_errors = [] 
    forecastng_errors = [] 
    
    for idx_d in xrange(kf_original_obj.max_it_BR):

        init_ = kf_original_obj.random_hyperparams_list[idx_randparams, :]
        truth = kf_original_obj.macro_truth[idx_randparams, idx_d, :]
        y_signal = truth + kf_original_obj.msmt_noise_variance*np.random.randn(truth.shape[0])

        np.savez(filepath_LS_Ensemble+'_BR_AKF_MAP', test='success')
        raise RuntimeError
        
        akf_prediction = akf(descriptor, y_signal, weights, 
                                init_[0], 
                                init_[1], 
                                n_train=kf_original_obj.n_train, 
                                n_testbefore=kf_original_obj.n_testbefore, 
                                n_predict=kf_original_obj.n_predict,
                                p0=10000, skip_msmts=skip_msmts)

        truth_ = truth_[kf_original_obj.n_train - kf_original_obj.n_testbefore : kf_original_obj.n_train + kf_original_obj.n_predict]
        residuals_sqr_errors = (akf_prediction.real - truth_.real)**2

        prediction_errors.append(residuals_sqr_errors[0: kf_original_obj.n_testbefore])
        forecastng_errors.append(residuals_sqr_errors[kf_original_obj.n_testbefore : ])
    
    macro_prediction_errors.append(prediction_errors)
    macro_forecastng_errors.append(forecastng_errors)

    np.savez(filepath_LS_Ensemble+'_BR_AKF_MAP',
                    msmt_noise_variance= kf_original_obj.msmt_noise_variance, 
                    weights=weights, 
                    max_it_BR = kf_original_obj.max_it_BR, 
                    num_randparams = kf_original_obj.num_randparams,
                    macro_truth=kf_original_obj.macro_truth,
                    skip_msmts = skip_msmts,
                    akf_macro_prediction_errors=macro_prediction_errors, 
                    akf_macro_forecastng_errors=macro_forecastng_errors,
                    random_hyperparams_list=kf_original_obj.random_hyperparams_list)