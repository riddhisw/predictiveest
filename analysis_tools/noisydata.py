#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 11:40:01 2017

@author: riddhisw
"""
from __future__ import division, print_function, absolute_import
import numpy as np

from analysis_tools.truth import Truth

class Noisy_Data(Truth):


    def __init__(self, msmt_noise_params, true_noise_params, user_defined_variance=None, **kwargs):

        Truth.__init__(self, true_noise_params, **kwargs)

        # Msmt Noise params
        self.msmt_noise_params = msmt_noise_params
        self.msmt_noise_mean = msmt_noise_params[0]
        self.msmt_noise_level = msmt_noise_params[1]
        self.user_defined_variance = user_defined_variance 
        self.msmt_noise_variance = None
        pass


    def msmt_noise_variance_calc(self, num_of_standard_dev=3.0):
        
        one_realisation = self.beta_z()[0]
        msmt_noise_var = self.msmt_noise_level*num_of_standard_dev*np.sqrt(np.var(one_realisation))
        #print one_realisation
        #print one_realisation[0]
        return msmt_noise_var # this is not variance. It's a new standard deviation.


    
    def generate_data_from_truth(self, user_defined_variance):

        if user_defined_variance !=None:
            # overrides existing msmt_noise_variance
            self.msmt_noise_variance = user_defined_variance
        
        if user_defined_variance==None and self.msmt_noise_variance == None:
            self.msmt_noise_variance = self.msmt_noise_variance_calc()

        if self.msmt_noise_variance != None:
            truth = self.beta_z()[0]
            y_signal =  truth + self.msmt_noise_variance*np.random.randn(self.number_of_points) # This is a standard deviation. True noise variance = msmt_noise_var **2 
            # Double check that np.random.normal(loc=0,scale=sdev) == sdev*np.random.randn()
            return  truth, y_signal

        return "boo"


