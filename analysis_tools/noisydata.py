#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
Created on Sat Apr  8 11:40:01 2017
@author: riddhisw

.. module:: analysis_tools.noisydata

    :synopsis: Simulates noisy experimental data based on a engineered true
        dephasing noise field.

    Module Level Classes:
    ----------------------
        Noisy_Data : Simulates noisy experimental data after inheriting
            from Truth class.

.. moduleauthor:: Riddhi Gupta <riddhi.sw@gmail.com>
'''

from __future__ import division, print_function, absolute_import
import numpy as np

from analysis_tools.truth import Truth

class Noisy_Data(Truth):
    ''' Simulates noisy experimental data after inheriting from Truth class.

    Attributes:
    ----------
        true_noise_params : Parameters to initiate Truth class.
        msmt_noise_params : Parameters to initiate Noisy_Data class, in order of:

            msmt_noise_mean (msmt_noise_params[0]) : True measurement noise mean.
            msmt_noise_level (msmt_noise_params[1]) : True measurement noise level.

        user_defined_variance (`float`, optional) : Manually overides strength of
            applied measurement noise via Noisy_Data.msmt_noise_variance_calc()
        msmt_noise_variance : Strength of applied measurement noise based on target
            true noise level in msmt_noise_params.

    Methods:
    -------
        msmt_noise_variance_calc() : Return msmt_noise_variance for applying noise
            to simulated data, based on Truth and target msmt_noise_level.
    '''

    def __init__(self, msmt_noise_params, true_noise_params, user_defined_variance=None, **kwargs):

        Truth.__init__(self, true_noise_params, **kwargs)

        # Msmt Noise params
        self.msmt_noise_params = msmt_noise_params
        self.msmt_noise_mean = msmt_noise_params[0]
        self.msmt_noise_level = msmt_noise_params[1]
        self.user_defined_variance = user_defined_variance
        self.msmt_noise_variance = None


    def msmt_noise_variance_calc(self, num_of_standard_dev=3.0):
        '''Return msmt_noise_variance for applying measurement noise to simulated
            experimental data, based on Truth and target msmt_noise_level.

        Parameters:
        ----------
            num_of_standard_dev (`int`, optional) : Links target measurement noise
                level as a expression of the variance of one realisation of random
                variables of a true, unobserved random process.

        Returns:
        -------
            msmt_noise_var : Standard deviation of applied measurement noise to
                simulate noisy data based on one realisation of true dephasing field.
        '''

        one_realisation = self.beta_z()[0]
        msmt_noise_var = self.msmt_noise_level*num_of_standard_dev*np.sqrt(np.var(one_realisation))

        return msmt_noise_var # this is not variance. It's a new standard deviation.



    def generate_data_from_truth(self, user_defined_variance):
        ''' Return one realisation of a true dephasing field and noisy
            simulated experimental data under this dephasing realisation.

        Parameters:
        ----------
            user_defined_variance (`float`, optional) : Manually overides strength of
            applied measurement noise via Noisy_Data.msmt_noise_variance_calc()

        Returns:
        -------
            truth (`float64`) :  One realisation of true dephasing field
                (ideal observations with no measurement noise).
            y_signal (`float64`) : Simulated (noisy) measurement data under one
                realisation of a true dephasing process, with observations
                    corrupted by white, Gaussian measurement noise.
        '''

        if user_defined_variance != None:
            # overrides existing msmt_noise_variance
            self.msmt_noise_variance = user_defined_variance

        if user_defined_variance is None and self.msmt_noise_variance is None:
            self.msmt_noise_variance = self.msmt_noise_variance_calc()

        if self.msmt_noise_variance != None:
            truth = self.beta_z()[0]
            y_signal =  truth + self.msmt_noise_variance*np.random.randn(self.number_of_points)
            # This is a standard deviation. True noise variance = msmt_noise_var **2
            # Double check that np.random.normal(loc=0,scale=sdev) == sdev*np.random.randn()
            return  truth, y_signal

        return "Unspecified ERROR in method generate_data_from_truth()"


