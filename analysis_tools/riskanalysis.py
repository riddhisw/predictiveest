'''
Created on Sat Apr  8 11:40:01 2017
@author: riddhisw

.. module:: analysis_tools.risk analysis

    :synopsis: Optimises LKFFB Kalman noise variance parameters for a parameter regime
        specified by (testcase, variation); and used Bayes Risk metric to assess
        predictive performance.

    Module Level Classes:
    ----------------------
        Bayes_Risk : Stores Bayes Risk map for a scenario specified by (testcase, variation).
        Create_KF_Experiment : Optimises KF filter parameters (sigma, R), and uses Bayes
            Risk metric for predictive power analysis.

.. moduleauthor:: Riddhi Gupta <riddhi.sw@gmail.com>
'''

from __future__ import division, print_function, absolute_import
import os
import numpy as np
# import time as t

from analysis_tools.kalman import Kalman
from analysis_tools.common import get_tuned_params_

class Bayes_Risk(object):
    ''' Stores Bayes Risk map for a scenario specified by (testcase, variation)

    Attributes:
    ----------
        bayes_params (`dtype`) : Parameters to intiate Bayes Risk class:
            max_it_BR (`int`) : Number of repetitions for a Bayes Risk calculation.
            num_randparams (`int`) : Number of random (sigma, R) sample pairs.
            space_size (`int`) : Exponent parameter to set orders of magnitude
                spanned by unknown noise variance parameters.
            truncation (`int`) : Pre-determined threshold for number of
                lowest input values to return in modcule function,
                get_tuned_params(), in module common.
        doparallel (`Boolean`) : Enable parallelisation of Bayes Risk calculations [DEPRECIATED].
        lowest_pred_BR_pair (`float64`) : (sigma, R) pair with min Bayes Risk in state estimation.
        lowest_fore_BR_pair (`float64`) : (sigma, R) pair with min Bayes Risk in prediction.
        means_list (`float64`) : Helper calculation for Bayes Risk.
        skip_msmts (`int`) : Number of time-steps to skip between measurements.
            To receive measurement at every time-step, set skip_msmts=1.
        did_BR_Map (`Boolean`) : When True, indicates that a BR Map has been created.
        macro_truth (`float64`) : Matrix data container for set of true noise realisations,
            generated for the purposes of calculating the Bayes Risk metric for all
            (sigma, R) random samples.
        macro_prediction_errors (`float64`) : Matrix data container for set of state estimates.
        macro_forecastng_errors (`float64`) : Matrix data containter for set of forecasts.
        random_hyperparams_list (`float64`) : Matrix data containter for random
            samples of (sigma, R).
    '''

    def __init__(self, bayes_params):
        '''Initiates a Bayes_Risk class instance. '''

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
    '''Optimises KF filter parameters (sigma, R), and uses Bayes Risk metric for
    predictive power analysis.

    Create_KF_Experiment class inherits from Bayes_Risk and Kalman class objects.

    Attributes:
    ----------
        bayes_params : Parameters for instantiating Bayes_Risk class object.
        *arge, **kwargs : Parameters for instantiating Kalman class object.
        filename_BR (`str`) : Filename for saving Bayes Risk output as .npz file.

    Methods:
    -------
        loss : Returns squared distance between Kalman state  and true dephasing.
        one_loss_trial : Return loss values for a specific choice of random samples (sigma, R),
            and a choice of skip_msmts.
        rand_param : Return a randomly sampled (sigma, R) pair.
        one_bayes_trial : Return true realisations, state etimation errors and
            prediction errors over max_it_BR repetitions for one (sigma, R) pair.
        naive_implementation : Return Bayes Risk analysis as a saved .npz file.
        func_x0 : Returns random dim=2 arrays from a parameter space defined by space_size.
        get_tuned_params : Helper function for Bayes Risk mapping.
        set_tuned_params : Helper function for Bayes Risk mapping.
    '''
    def __init__(self, bayes_params, *args, **kwargs):
        '''Initiates a Create_KF_Experiment class instance.'''

        Bayes_Risk.__init__(self, bayes_params)
        Kalman.__init__(self, *args, **kwargs)
        self.filename_BR = self.filename0+str('BR_Map')


    def loss(self, init_, y_signal_, skip_msmts_, truth_):
        ''' Return squared distance between Kalman state predictions and true dephasing.

        Parameters:
        ----------
            init_ : List containing Kalman noise variances [sigma, R] for filtering.
            y_signal_ (`float64`) :  Noisy measurement data (input to filtering).
            skip_msmts_ (`int`) : Number of time-steps to skip between measurements.
                To receive measurement at every time-step, set skip_msmts=1.
            truth_ : One realisation of true dephasing field (ideal observations
                with no measurement noise).

        Returns:
        -------
            residuals_sqr_errors : a sequence of squared errors between algorithm
                output and engineered truths.

        '''

        # tricky if Kalman output yields more than just prediction sequence
        predictions = self.single_prediction(y_signal_, skip_msmts_, init=init_)
        truth_ = truth_[self.n_train - self.n_testbefore : self.n_train + self.n_predict]
        residuals_sqr_errors = (predictions.real - truth_.real)**2

        return residuals_sqr_errors # not summed over prediction steps


    def one_loss_trial(self, random_hyperparams, skip_msmts):
        ''' Return loss values for a specific choice of random samples (sigma, R),
        and a choice of skip_msmts.

        Parameters:
        ----------
            random_hyperparams (`float64`) : A single random pair of [sigma, R].
            skip_msmts (`int`) : Number of time-steps to skip between measurements.
                To receive measurement at every time-step, set skip_msmts=1.

        Returns:
        -------
            Output[0] (`float64`) : True dephasing field generated by generate_data_from_truth()
            Output[1] (`float64`) : Kalman state estimation errors.
            Output[2] (`float64`) : Kalman prediction errors.
        '''

        truth, y_signal = self.generate_data_from_truth(self.user_defined_variance)
        errors = self.loss(random_hyperparams, y_signal, skip_msmts, truth)

        return truth, errors[0:self.n_testbefore], errors[self.n_testbefore : self.n_testbefore + self.n_predict]


    def rand_param(self):
        ''' Return a randomly sampled (sigma, R) pair. '''
        return np.array([self.func_x0(), self.func_x0()])


    def one_bayes_trial(self, _):
        ''' Return true realisations, state etimation errors and prediction errors
        over max_it_BR repetitions for one (sigma, R) pair. '''

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
        ''' Return Bayes Risk analysis as a saved .npz file over max_it_BR
        repetitions of true dephasing noise and simulated datasets; for
        num_randparams number of random (sigma, R) pairs.

        Parameters:
        ----------
            change_skip_msmts (`int`, optional) : Manually specify skip_msmts.
                Defaults to 1 (all measurements considered).BaseException
        Returns:
        -------
            Output .npz file containing all Bayes Risk data for analysis.
        '''

        if change_skip_msmts != 1:
            self.skip_msmts = change_skip_msmts
            print("Skipped Msmts Changed from %s to %s" %(1, self.skip_msmts))

        self.random_hyperparams_list = []
        self.macro_prediction_errors = []
        self.macro_forecastng_errors = []
        self.macro_truth = []

        # start_outer_multp = t.time()

        for ind in xrange(self.num_randparams):

            full_bayes_map = self.one_bayes_trial(None)

            self.macro_truth.append(full_bayes_map[0])
            self.macro_prediction_errors.append(full_bayes_map[1])
            self.macro_forecastng_errors.append(full_bayes_map[2])
            self.random_hyperparams_list.append(full_bayes_map[3])

        # total_outer_multp = t.time() - start_outer_multp

        np.savez(os.path.join(self.savetopath, self.filename_BR),
                 end_run=ind,
                 filname0=self.filename0,
                 max_it=self.max_it,
                 savetopath=self.savetopath,
                 expt_params=self.expt_params,
                 kalman_params=self.kalman_params,
                 msmt_noise_params=self.msmt_noise_params,
                 true_noise_params=self.true_noise_params[1:],
                 true_signal_params=self.true_signal_params[1:],
                 user_defined_variance=[self.user_defined_variance],
                 msmt_noise_variance=self.msmt_noise_variance,
                 max_it_BR=self.max_it_BR,
                 num_randparams=self.num_randparams,
                 spacesize=self.space_size,
                 macro_truth=self.macro_truth,
                 skip_msmts=self.skip_msmts,
                 macro_prediction_errors=self.macro_prediction_errors,
                 macro_forecastng_errors=self.macro_forecastng_errors,
                 random_hyperparams_list=self.random_hyperparams_list)

        self.did_BR_Map = True


    def func_x0(self):
        '''
        Returns random dim =2 arrays from a parameter space defined by space_size
        [Helper function for Bayes Risk mapping]
        '''
        maxindex = self.space_size.shape[0]-1
        ind = int(np.random.uniform(low=0, high=maxindex))
        exponent = self.space_size[ind]
        return np.random.uniform(0,1)*(10.0**exponent)


    def get_tuned_params(self, max_forecast_loss):
        '''[Helper function for Bayes Risk mapping]'''
        self.means_lists_, self.lowest_pred_BR_pair, self.lowest_fore_BR_pair = get_tuned_params_(max_forecast_loss,
                                                                                                  np.array(self.num_randparams),
                                                                                                  np.array(self.macro_prediction_errors),
                                                                                                  np.array(self.macro_forecastng_errors),
                                                                                                  np.array(self.random_hyperparams_list),
                                                                                                  self.truncation)

    def set_tuned_params(self):
        '''[Helper function for Bayes Risk mapping]'''
        self.optimal_sigma = self.lowest_pred_BR_pair[0]
        self.optimal_R = self.lowest_pred_BR_pair[1]
