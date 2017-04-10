# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
import numpy as np
import time as t

from  ML_Kalman import Kalman

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


    def loss(self, init_, y_signal_, truth_):
        
        # tricky if Kalman output yields more than just prediction sequence
        predictions = self.single_prediction(y_signal_, init=init_)
        truth_ = truth_[self.n_train - self.n_testbefore : self.n_train + self.n_predict]
        residuals_sqr_errors = (predictions.real - truth_.real)**2
        
        return residuals_sqr_errors # not summed over prediction steps


    def one_loss_trial(self, random_hyperparams):
        
        truth, y_signal = self.generate_data_from_truth(self.user_defined_variance)
        errors = self.loss(random_hyperparams, y_signal, truth)
        
        return truth, errors[0:self.n_testbefore], errors[self.n_testbefore : self.n_testbefore + self.n_predict]


    def rand_param(self):
        return np.array([self.func_x0(), self.func_x0()]) 


    def one_bayes_trial(self, _):
        
        
        init_ = self.rand_param()
        
        prediction_errors = [] 
        forecastng_errors = [] 
        truths_in_trials = [] 

        for ind in xrange(self.max_it_BR):
        
            true, pred, fore = self.one_loss_trial(init_)
            
            truths_in_trials.append(true)
            prediction_errors.append(pred)
            forecastng_errors.append(fore)

        return truths_in_trials, prediction_errors, forecastng_errors, init_
    
    
    def naive_implementation(self):

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


    def truncate_losses(self, list_of_loss_vals):
        '''
        Returns truncation number of hyperparameters for lowest risk from a sequence of outcomes.
        [Helper function for Bayes Risk mapping]
        '''
        
        loss_index_list = list(enumerate(list_of_loss_vals))
        low_loss = sorted(loss_index_list, key=lambda x: x[1])
        indices = [x[0] for x in low_loss]
        losses = [x[1] for x in low_loss]
        return indices[0:self.truncation], losses[0:self.truncation]

        
    def get_tuned_params(self):
        '''
        Returns optimal sigma, R based on lowest prediction and forecasting losses. 
        '''
        
        if self.did_BR_Map == True:

            prediction_errors_stats = np.zeros((self.num_randparams, 2)) 
            forecastng_errors_stats = np.zeros((self.num_randparams, 2)) 
            
            j=0
            for j in xrange(self.num_randparams):
                
                prediction_errors_stats[ j, 0] = np.mean(self.macro_prediction_errors[j])
                prediction_errors_stats[ j, 1] = np.var(self.macro_prediction_errors[j])
                forecastng_errors_stats[ j, 0] = np.mean(self.macro_forecastng_errors[j])
                forecastng_errors_stats[ j, 1] = np.var(self.macro_forecastng_errors[j])     
            
            means_list =  prediction_errors_stats[:,0] 
            means_list2 = forecastng_errors_stats[:,0]
            self.means_lists_= [means_list, means_list2]
    
            x_data, y_data = self.truncate_losses(means_list)
            x2_data, y2_data = self.truncate_losses(means_list2)
            
            index1 = int(x_data[0])
            index2 = int(x_data[1])
    
            self.lowest_pred_BR_pair = self.random_hyperparams_list[index1]
            self.lowest_fore_BR_pair = self.random_hyperparams_list[index2]
            
            print("Optimal params fore prediction and forecasting", self.lowest_pred_BR_pair, self.lowest_fore_BR_pair)
            pass

        elif self.did_BR_Map == False:
            print('BR MAP not implemented in this instance. No tuned parameters calc possible. Use different class to load and analyse BR Map Data.' )
        pass
    
    def set_tuned_params(self):
        self.optimal_sigma = self.lowest_pred_BR_pair[0]
        self.optimal_R = self.lowest_pred_BR_pair[1]
        pass