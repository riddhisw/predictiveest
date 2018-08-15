'''
Created on Thu Apr 20 19:20:43 2017
@author: riddhisw

.. module:: data_tools.common

    :synopsis: List of functions required by multiple modules in data_tools.

    Module Level Functions:
    ----------------------
        get_data :  Return noisy measurements given a randomly chosen realisation of truth.

.. moduleauthor:: Riddhi Gupta <riddhi.sw@gmail.com>

'''
from __future__ import division, print_function, absolute_import
import numpy as np

def get_data(dataobject):

    '''Return noisy measurements given a randomly chosen realisation of truth.

    Pick one realisations of truths in LKFFB cluster data file denoted by
    (test_case, variation) and return noisy msmts, as inputs to any other algorithm.
    (Faster than generating another true noise realisation). A randomly chosen
    realisation of truth is picked from macro_truths dataobject - a LoadExperiment
    instance for LKFFB for given (test_case, variation) pair.
    '''

    msmt_noise_variance = dataobject.LKFFB_msmt_noise_variance
    number_of_points = dataobject.Expt.number_of_points
    n_predict = dataobject.Expt.n_predict
    shape = dataobject.LKFFB_macro_truth.shape

    macro_truth = dataobject.LKFFB_macro_truth.reshape(shape[0]*shape[1], shape[2]) 
    # collapse  first two axees (only relevant to KF techniques)

    pick_one = int(np.random.uniform(low=0, high = int(macro_truth.shape[0]-1)))

    return macro_truth[pick_one, :] + msmt_noise_variance*np.random.randn(number_of_points), pick_one
