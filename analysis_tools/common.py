#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 19:20:43 2017

@author: riddhisw
"""

from __future__ import division, print_function, absolute_import
import numpy as np

def truncate_losses_( list_of_loss_vals, truncation):
    '''
    Returns truncation number of hyperparameters for lowest risk from a sequence of outcomes.
    [Helper function for Bayes Risk mapping]
    '''
    
    loss_index_list = list(enumerate(list_of_loss_vals))
    low_loss = sorted(loss_index_list, key=lambda x: x[1])
    indices = [x[0] for x in low_loss]
    losses = [x[1] for x in low_loss]
    return indices[0:truncation], losses[0:truncation]


def get_tuned_params_(max_forecast_loss, num_randparams, 
                      macro_prediction_errors, macro_forecastng_errors, 
                      random_hyperparams_list):
    
    prediction_errors_stats = np.zeros((num_randparams, 2)) 
    forecastng_errors_stats = np.zeros((num_randparams, 2)) 
    
    j=0
    for j in xrange(num_randparams):
        
        prediction_errors_stats[ j, 0] = np.mean(macro_prediction_errors[j])
        prediction_errors_stats[ j, 1] = np.var(macro_prediction_errors[j])
        forecastng_errors_stats[ j, 0] = np.mean(macro_forecastng_errors[j, :, 0:max_forecast_loss]) 
        forecastng_errors_stats[ j, 1] = np.var(macro_forecastng_errors[j, :, 0:max_forecast_loss])     
    
    means_list =  prediction_errors_stats[:,0] 
    means_list2 = forecastng_errors_stats[:,0]
    means_lists_= [means_list, means_list2]

    x_data, y_data = truncate_losses_(means_list)
    x2_data, y2_data = truncate_losses_(means_list2)

    lowest_pred_BR_pair = random_hyperparams_list[x_data[0], :]
    lowest_fore_BR_pair = random_hyperparams_list[x2_data[0], :]
    
    return means_lists_, lowest_pred_BR_pair, lowest_fore_BR_pair