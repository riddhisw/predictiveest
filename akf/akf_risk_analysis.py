"""
Created on Thu Apr 20 19:20:43 2017

@author: riddhisw

PACKAGE: akf
MODULE: akf.akf_risk_analysis

The purpose of akf package is to implement autoregressive Kalman Filtering (AKF).

We use a Bayes Risk Map from LKFFB, and the same choices of random 
hyperparameters (sigma, R) and true noise realisations to create
the equivalent optimisation problem for an autoregressive Kalman Filter with 
LS Filter weights.

MODULE PURPOSE: Optimises an AKF filter for Kalman design parameters (sigma, R)

CLASSES: AKF_Optimisation
METHODS: AKF_Optimisation.Method()
make_BR_AKF_MAP -- Returns a Bayes Risk map for random (sigma, R) pairs using AKF

"""
from __future__ import division, print_function, absolute_import

import sys
import numpy as np
from data_tools.load_raw_cluster_data import LoadExperiment as le

from analysis_tools.common import sqr_err
from data_tools.common import get_data
from akf.common import fetch_weights
from akf.armakf import autokf as akf

class AKF_Optimisation(object):
    

    def __init__(self, test_case, variation, 
                 LKFFBfilepath, LSFfilepath, AKF_savetopath,
                 skip_msmts=1, choose_step_fwds=1):
        
        self.test_case = test_case 
        self.variation = variation  
        self.LKFFBfilepath = LKFFBfilepath 
        self.LSFfilepath = LSFfilepath  
        self.AKF_savetopath = AKF_savetopath

        self.dataobject = le(self.test_case, self.variation,
                             GPRP_load='No', 
                             LKFFB_load = 'Yes', LKFFB_path = self.LKFFBfilepath,
                             AKF_load='No',
                             LSF_load = 'Yes', LSF_path=self.LSFfilepath)

        self.AKF_skip_msmts = skip_msmts 
        self.AKF_choose_step_fwds = choose_step_fwds
        self.AKF_weights = fetch_weights(self.dataobject)

        pass


    def make_BR_AKF_MAP(self, mapname='_BR_AKF_MAP_correctQ_'):
        ''' Returns a Bayes Risk map for random (sigma, R) pairs using AKF
        '''

        path2dir = self.AKF_savetopath+'test_case_'+str(self.test_case)+'_var_'+str(self.variation)

        macro_prediction_errors = [] 
        macro_forecastng_errors = [] 
        store_macro_truths = [] # Should be exactly equal to self.dataobject.LKFFB_macro_truth if code is correcty

        for idx_randparams in xrange(self.dataobject.LKFFB_num_randparams):

            prediction_errors = [] 
            forecastng_errors = []
            store_this_truth = []
            
            for idx_d in xrange(self.dataobject.LKFFB_max_it_BR):

                init_ = self.dataobject.LKFFB_random_hyperparams_list[idx_randparams, :]
                truth = self.dataobject.LKFFB_macro_truth[idx_randparams, idx_d, :]
                y_signal = truth + self.dataobject.LKFFB_msmt_noise_variance*np.random.randn(truth.shape[0])
                
                akf_prediction = akf(path2dir+'_AKF_', y_signal, self.AKF_weights, 
                                     init_[0], 
                                     init_[1], 
                                     n_train=self.dataobject.Expt.n_train, 
                                     n_testbefore=self.dataobject.Expt.n_testbefore, 
                                     n_predict=self.dataobject.Expt.n_predict,
                                     p0=10000, skip_msmts=self.AKF_skip_msmts)

                truth_ = truth[self.dataobject.Expt.n_train - self.dataobject.Expt.n_testbefore : self.dataobject.Expt.n_train + self.dataobject.Expt.n_predict]
                residuals_sqr_errors = sqr_err(akf_prediction, truth_) ## (akf_prediction.real - truth_.real)**2

                prediction_errors.append(residuals_sqr_errors[0: self.dataobject.Expt.n_testbefore])
                forecastng_errors.append(residuals_sqr_errors[self.dataobject.Expt.n_testbefore : ])
                store_this_truth.append(truth)
            
            macro_prediction_errors.append(prediction_errors)
            macro_forecastng_errors.append(forecastng_errors)
            store_macro_truths.append(store_this_truth)

            np.savez(path2dir+mapname,
                     msmt_noise_variance= self.dataobject.LKFFB_msmt_noise_variance, 
                     weights= self.AKF_weights, 
                     max_it_BR = self.dataobject.LKFFB_max_it_BR,
                     num_randparams = self.dataobject.LKFFB_num_randparams,
                     macro_truth=store_macro_truths, 
                     skip_msmts = self.AKF_skip_msmts,
                     akf_macro_prediction_errors=macro_prediction_errors, 
                     akf_macro_forecastng_errors=macro_forecastng_errors,
                     random_hyperparams_list=self.dataobject.LKFFB_random_hyperparams_list)