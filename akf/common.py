'''
Created on Thu Apr 20 19:20:43 2017

@author: riddhisw

PACKAGE: akf
MODULE: akf.common

The purpose of akf package is to implement autoregressive Kalman Filtering (AKF).

MODULE PURPOSE: Simplies the AKF optimisation problem by using autoregressive weight 
estimates from the LSF.

METHODS: 
fetch_weights -- Returns LSF weights to approximate the dynamical model in AKF  
'''

from __future__ import division, print_function, absolute_import
import numpy as np

def fetch_weights(dataobject, choose_stp_fwds=1):
    ''' Returns LSF weights to approximate the dynamical model in AKF  
    '''

    LSF_mean_weights = np.mean(dataobject.LSF_macro_weights[:,:,:,0],  axis=0)  # consistent with LSF approach in DATA_v0, not DRAFT_1
    # choose_stp_fwds = 1 implies we choose the best model, i.e. the one step ahead weights in AR(q) LSF
    AKF_chosen_weights = LSF_mean_weights[choose_stp_fwds, :] 
    return AKF_chosen_weights
