#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 19:20:43 2017

@author: riddhisw
"""

from __future__ import division, print_function, absolute_import
import numpy as np


def sqr_err(predictions, truth):
    ''' Returns the squared error sequence between predictions sequence and truth sequence
    '''
    return (predictions.real - truth.real)**2
    

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
                      random_hyperparams_list, truncation):
    
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

    x_data, y_data = truncate_losses_(means_list, truncation)
    x2_data, y2_data = truncate_losses_(means_list2, truncation)

    lowest_pred_BR_pair = random_hyperparams_list[x_data[0], :]
    lowest_fore_BR_pair = random_hyperparams_list[x2_data[0], :]
    
    return means_lists_, lowest_pred_BR_pair, lowest_fore_BR_pair


#########################
# LS ANALYSIS
#########################

def calc_LS_amps(macro_weights):

    '''Returns amplitude estimates from LS macro_weights
        n_predict - max number of steps forward
        past_msmts - total number of past msmts in LS model
        macro_weights - [Dims: ensemble_size x n_predict x past_msmts x 1]

        Each n_predict x past_msmts matrix of weights is obtained by 
        giving the LS filter n_train data points. We then run the filter n_predict times,
        where each time we get an N-step ahead prediction, N = 0, 1, 2, ... n_predict 

        Hence macro_weights[0, n_predict, past_msmts, 0] represents weights for 
        underlying time separations (in discrete steps) of:

        [[ 0, 1, 2, .... past_msmts -1*],
        [ 1, 2, 3, .... past_msmts   ],
        [ 2, 3, 4, .... past_msmts +1],
        ...,
        [ n_predict -1*, n_predict, n_predict+1, .... past_msmts + n_predict - 2*]]

        *NB: The extra '-1' and '-2' comes from Python indexing starting at zero

        We want to add (or take the mean) of weights which are effectively wieghting
        the same time separation. These are off-diagonal elements of a rectangular
        matrix. However, this is easier to do if we flip macro_weights matrix (below)

        If we take the mean across each set of off-diagonals, we are taking 
        an expectation value based on a single validation data set but different models. 

        If we take the mean across zero-th axis of macro_weights, we are taking 
        an expectation value over many validation data sets.

        topmost row of weights == step_fwds = 0
        bottom row of weights == step_fwds = 50 '''
    

    flipped = np.fliplr(np.mean(macro_weights, axis=0)[:,:,0]) # ensemble avg
    amps = calc_mean_off_diagonals(flipped)    
    
    return amps 



def calc_mean_off_diagonals(flipped_LS_weights_matrix):
    '''Returns mean of off diagonal elements of a rectangular matrix. 
    We assume dim(flipped_weights_matrix) = stps_fwd x past_msmts,
    and past_msmts >> stps_fwd. 
    
    [TO DO: error proof code for past_msmts <= stps_fwd]'''

    stps_fwd = flipped_LS_weights_matrix.shape[0] 
    past_msmts = flipped_LS_weights_matrix.shape[1]

    amps = np.zeros(stps_fwd + past_msmts -1)
    edge_index = past_msmts -1

    # get off-diagonals along first axis
    for stpfwd in xrange(stps_fwd):
        amps[edge_index+stpfwd] = np.mean(flipped_LS_weights_matrix.diagonal(-stpfwd)) # model avg
    
    # get off-diagonals along second axis
    for longaxis in xrange(past_msmts):
        amps[edge_index-longaxis] = np.mean(flipped_LS_weights_matrix.diagonal(longaxis))  # model avg

    return amps


def calc_periodogram(x_p, Delta_S_Sampling, Delta_T_Sampling):
    ''' Calculates periodogram of time domain signal, x_p, based on sampling rates.
    '''
    
    number_of_points = int(1.0/(Delta_S_Sampling*Delta_T_Sampling))
    halfN = int((number_of_points - 1)/ 2.0)

    S_est = np.zeros(halfN)
    omega = np.zeros(halfN)

    for idx_j in range(1, halfN, 1):
        omega_j = 2.0*np.pi*idx_j*Delta_S_Sampling

        one_j_term = 0
        for idx_t in range(0, number_of_points, 1):
            one_j_term += x_p[idx_t]*np.exp(-1.0j*idx_t*Delta_T_Sampling*omega_j)

        S_est[idx_j] = (1.0/(2.0*np.pi*number_of_points))*abs(one_j_term)**2
        omega[idx_j] = omega_j
    
    return omega, S_est
 
    
def calc_AR_PSD(phi_p, sigma_n_sqr, Delta_S_Sampling, Delta_T_Sampling):
    ''' Returns power spectral density S(w) and w for an autogressive, covariance stationary process of order p.
    REF:
    Lecture Notes (http://www.lmd.ens.fr/E2C2/class/SASP_lecture_notes.pdf) 
    Wikipedia (https://en.wikipedia.org/wiki/Autoregressive_model#Spectrum)
    '''
    
    number_of_points = 1.0/(Delta_S_Sampling*Delta_T_Sampling)
    halfN = int((number_of_points - 1) / 2.0)
    order_p = phi_p.shape[0]

    S_thr = np.zeros(halfN)
    omega = np.zeros(halfN)
    
    for idx_j in range(1, halfN, 1):

        omega_j = 2.0*np.pi*idx_j*Delta_S_Sampling

        one_j_term = 0
        for idx_p in range(0, order_p, 1):
            one_j_term += phi_p[idx_p]*np.exp(-1.0j*(idx_p + 1)*Delta_T_Sampling*omega_j)

        S_thr[idx_j] = sigma_n_sqr/(abs(1 - one_j_term)**2)
        omega[idx_j] = omega_j
    
    return omega, S_thr