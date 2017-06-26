'''
Created on Thu Apr 20 19:20:43 2017

@author: riddhisw

PACKAGE: data_tools
MODULE: data_tools.common

The purpose of data_tools is to load data and analyse data generated 
by any algorithm (LKFFB, AKF, LSF, GPy) for any scenario (test_case, variation)

MODULE PURPOSE: Picks one of 75*50 realisations of truths in LKFFB cluster data file 
denoted by (test_case, variation) and returns noisy msmts as inputs to any other algorithm 

METHODS: 
get_data -- Returns noisy msmts given a randomly chosen realisation of truth

'''
from __future__ import division, print_function, absolute_import
import numpy as np

def get_data(dataobject):
    
    ''' 
    Returns noisy msmts given a randomly chosen realisation of truth from macro_truths.
    dataobject --- LoadExperiment instance for LKFFB for given test_case, and variation. 
    '''

    msmt_noise_variance = dataobject.LKFFB_msmt_noise_variance
    number_of_points = dataobject.Expt.number_of_points
    n_predict = dataobject.Expt.n_predict
    shape = dataobject.LKFFB_macro_truth.shape
    macro_truth = dataobject.LKFFB_macro_truth.reshape(shape[0]*shape[1], shape[2]) # collapse  first two axees (only relevant to KF techniques)
    pick_one = int(np.random.uniform(low=0, high = int(macro_truth.shape[0]-1)))
    
    return macro_truth[pick_one, :] + msmt_noise_variance*np.random.randn(number_of_points), pick_one