# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import numpy as np
import time as t

from analysis_tools.kalman import Kalman
from analysis_tools.common import get_tuned_params_

class Bayes_Risk(object):
    
   
    def __init__(self, bayes_params):

        self.max_it_BR = bayes_params[0]
        self.num_randparams = bayes_params[1]
        self.space_size = bayes_params[2]
        self.doparallel = True
        self.bayes_params = bayes_params
        self.truncation = bayes_params[3]
        self.lowest_pred_BR_pair = None
        self.lowest_fore_BR_pair = None
        self.means_list = None
        self.skip_msmts = 1
        
        self.did_BR_Map = False
        self.macro_truth = None
        self.macro_prediction_errors = None
        self.macro_forecastng_errors = None
        self.random_hyperparams_list = None
        pass

        
class Create_KF_Experiment(Bayes_Risk, Kalman):
    
    
    def __init__(self, bayes_params, *args, **kwargs):
        
        Bayes_Risk.__init__(self, bayes_params)
        Kalman.__init__(self, *args, **kwargs)

        self.filename_BR = self.filename0+str('BR_Map')
    
        pass


    def loss(self, init_, y_signal_, skip_msmts_, truth_):
        
        # tricky if Kalman output yields more than just prediction sequence
        predictions = self.single_prediction(y_signal_, skip_msmts_, init=init_)
        truth_ = truth_[self.n_train - self.n_testbefore : self.n_train + self.n_predict]
        residuals_sqr_errors = (predictions.real - truth_.real)**2
        
        return residuals_sqr_errors # not summed over prediction steps


    def one_loss_trial(self, random_hyperparams, skip_msmts):
        
        truth, y_signal = self.generate_data_from_truth(self.user_defined_variance)
        errors = self.loss(random_hyperparams, y_signal, skip_msmts, truth)
        
        return truth, errors[0:self.n_testbefore], errors[self.n_testbefore : self.n_testbefore + self.n_predict]


    def rand_param(self):
        return np.array([self.func_x0(), self.func_x0()]) 


    def one_bayes_trial(self, _):
        
        
        init_ = self.rand_param()
        skip_msmts_ = self.skip_msmts
        
        prediction_errors = [] 
        forecastng_errors = [] 
        truths_in_trials = [] 

        for ind in xrange(self.max_it_BR):
        
            true, pred, fore = self.one_loss_trial(init_, skip_msmts_)
            
            truths_in_trials.append(true)
            prediction_errors.append(pred)
            forecastng_errors.append(fore)

        return truths_in_trials, prediction_errors, forecastng_errors, init_
    
    
    def naive_implementation(self, change_skip_msmts=1):
        
        if change_skip_msmts != 1:
            self.skip_msmts = change_skip_msmts
            print("Skipped Msmts Changed from %s to %s" %(1, self.skip_msmts))

        self.random_hyperparams_list = []
        self.macro_prediction_errors = [] 
        self.macro_forecastng_errors = [] 
        self.macro_truth = []

        start_outer_multp = t.time()        

        for ind in xrange(self.num_randparams):

            full_bayes_map = self.one_bayes_trial(None)

            self.macro_truth.append(full_bayes_map[0])
            self.macro_prediction_errors.append(full_bayes_map[1])
            self.macro_forecastng_errors.append(full_bayes_map[2])
            self.random_hyperparams_list.append(full_bayes_map[3])

        total_outer_multp = t.time() - start_outer_multp
        
        print( "Time Taken for BR Map: ", total_outer_multp)
        
        np.savez(os.path.join(self.savetopath, self.filename_BR),
                 end_run=ind, filname0=self.filename0, max_it = self.max_it, savetopath=self.savetopath,
                 expt_params = self.expt_params, 
                 kalman_params= self.kalman_params, 
                 msmt_noise_params= self.msmt_noise_params, 
                 true_noise_params= self.true_noise_params[1:], 
                 true_signal_params= self.true_signal_params[1:],
                 user_defined_variance= [self.user_defined_variance],
                 msmt_noise_variance= self.msmt_noise_variance,
                 max_it_BR = self.max_it_BR, num_randparams = self.num_randparams,
                 spacesize = self.space_size,
                 macro_truth=self.macro_truth,
                 skip_msmts = self.skip_msmts,
                 macro_prediction_errors=self.macro_prediction_errors, 
                 macro_forecastng_errors=self.macro_forecastng_errors,
                 random_hyperparams_list=self.random_hyperparams_list)
        
        self.did_BR_Map = True
        pass


    def func_x0(self):
        '''
        Returns random dim =2 arrays from a parameter space defined by space_size
        [Helper function for Bayes Risk mapping]
        '''
        maxindex = self.space_size.shape[0]-1
        ind = int(np.random.uniform(low=0, high=maxindex))
        exponent = self.space_size[ind]
        return np.random.uniform(0,1)*(10**exponent)


    def get_tuned_params(self, max_forecast_loss):
        self.means_lists_, self.lowest_pred_BR_pair, self.lowest_fore_BR_pair = get_tuned_params_(max_forecast_loss,
                                                                                                  num_randparams=self.num_randparams, 
                                                                                                  macro_prediction_errors=self.macro_prediction_errors, 
                                                                                                  macro_forecastng_errors=self.macro_forecastng_errors,
                                                                                                  random_hyperparams_list=self.random_hyperparams_list)
        print("Optimal params", self.lowest_pred_BR_pair, self.lowest_fore_BR_pair)
        pass


    
    def set_tuned_params(self):
        self.optimal_sigma = self.lowest_pred_BR_pair[0]
        self.optimal_R = self.lowest_pred_BR_pair[1]
        pass