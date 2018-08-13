#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
Created on Thu Apr 20 19:20:43 2017
@author: riddhisw

.. module:: analysis_tools.kalman

    :synopsis: Implements a Livska Kalman Filter using Fixed Basis (LKFFB).

    Module Level Classes:
    --------------------
        Kalman : Execute LKFFB runs for a given Experiment and Noisy Data regime.

.. moduleauthor:: Riddhi Gupta <riddhi.sw@gmail.com>

'''

from __future__ import division, print_function, absolute_import

import os
import numpy as np
import matplotlib.pyplot as plt
plt.ioff()

import sys
sys.path.append("../") # Look in the parent directory containing both kf and analysis_tools packages
from kf import fast as skf
from kf import detailed as dkf

from analysis_tools.experiment import Experiment
from analysis_tools.noisydata import Noisy_Data
from analysis_tools.common import sqr_err

FUDGE = 0.5 # Scaling factor for debugging and plotting [DEPRECIATED].
HILBERT_TRANSFORM = 2.0

class Kalman(Experiment, Noisy_Data):
    ''' Execute LKFFB runs for a given Experiment and Noisy Data regime.

    Attributes:
    ----------
        max_it (`int`) : Number of iterations of an LKKFB run.
        expt_params : Parameters for Experiment class.
        msmt_noise_params : Parameters for Noisy_Data class.
        true_noise_params : Parameters for Noisy_Data class.
        kalman_params : Parameters for LKKFB in Kalman class, in order of:
            optimal_sigma (`float64`) : Kalman process noise variance scale.
            optimal_R (`float64`) : Kalman measurement noise variance scale.
            x0 (`float64`) : Kalman initial state for all sub-states.
            p0 (`float64`) :  Kalman initial state variance for all sub-states.
            bdelta (`float64`) : LKFFB spacing between oscillators. (Hz).
            bmax (`int`) : Max frequency in LKKFB basis. (Hz).
        filename0 (`str`) : Parameter for filename for .npz output.
        filename_KF (`str`) :  Parameter for specifying filename for .npz output.
        savetopath (`type`) :  Parameter for specifying filepath for .npz output.
        basisA (`float64`) : Built-in Fourier basis, equidistanct frequencies.
        basisB (`float64`) : Built-in Fourier basis, limited lower cut-off at Fourier resolution.
        basisC (`float64`) : Built-in Fourier basis, as basisB but allows projection on zero basis frequency.
        basis_list (`list`) : List of built-in calling options for basisA, basiB, basisC. 
        phase_choice (`float`) : Phase correction for basisB or basisC.
        basis_dict (`type`) : List of built-in basis choices: basisA, basiB, basisC.
        phase_dict (`type`) : List of built-in phase corrections for choice of  basisA, basiB or basisC.
        prediction_method_list (`type`) : Choice of prediction / forecasting method,
            once data collection ceases. Defaults to 'ZeroGain'.

    Methods:
    -------
        calc_phase_correction : Calculate phase correction for choice of LKFFB basis, else returns zero.
        single_prediction : Return predictions for LKFFB (using ZeroGain).
        detailed_single_prediction : Return predictions and inst. amplitudes for LKFFB (using PropForward).
        ensemble_avg_predictions : Return ensemble averaged LKFFB prediction over Basis A,B,C and Prediction Methods.
        convert_amp_hz_to_radians : Return PSD in radians based on a frequency axis.
        run_test_KF : Compare output across LKFFB variants for given parameters [DEPRECIATED].


    '''

    def __init__(self, filename0, savetopath, max_it, expt_params, kalman_params,
                 msmt_noise_params, true_noise_params, user_defined_variance=None):

        Experiment.__init__(self, expt_params)

        if user_defined_variance == None:
            Noisy_Data.__init__(self, msmt_noise_params, true_noise_params, user_defined_variance=None)
        if user_defined_variance != None:
            Noisy_Data.__init__(self, msmt_noise_params, true_noise_params, user_defined_variance)

        self.max_it = max_it
        #self.skip_msmts = 1

        # Kalman Params
        self.kalman_params = kalman_params
        self.optimal_sigma = kalman_params[0]
        self.optimal_R = kalman_params[1]
        self.x0 = kalman_params[2]
        self.p0 = kalman_params[3]
        self.bdelta = kalman_params[4] # 1.0/(self.Delta_T_Sampling*self.n_train)

        self.bmax = self.bandwidth

        # Filenames for saving data
        self.filename0 = filename0
        self.filename_KF = self.filename0+'_kfresults'
        self.savetopath = savetopath

        # Pre-defined choices for computational basis
        self.basisA = np.arange(0.0, self.bmax, self.bdelta)
        self.basisB = np.arange(self.Delta_S_Sampling, self.bmax, self.bdelta)
        self.basisC = np.zeros(len(self.basisB)+1)
        self.basisC[1:] = self.basisB

        self.basis_list = ['A', 'B', 'C']
        self.phase_choice = self.calc_phase_correction(self.bdelta, self.Delta_S_Sampling, 'Yes')
        self.basis_dict = {'A': self.basisA, 'B': self.basisB, 'C': self.basisC}
        self.phase_dict = {'A': 0.0, 'B': self.phase_choice, 'C': self.phase_choice}

        # Pre-defined choices for prediction methods
        self.prediction_method_list = ["ZeroGain", "PropForward"]


    def calc_phase_correction(self, bdelta, Delta_S_Sampling, phase_correction):
        ''' Calculate phase correction for choice of LKFFB basis, else returns zero.

        Parameters:
        ----------
            bdelta : Spacing between adajcent basis frequencies in LKFFB.
            Delta_S_Sampling : Fourier domain resolution set by experimental
                sampling rate and number of measurements.
            phase_correction : Phase correction due to choice of Basis B or C in LKFFB.

        Returns:
        ------
            phase_correction : Phase adjustment to state prediction via
            PropForward under Basis B or C.
        '''

        phase_correction = 0.0
        if phase_correction == 'Yes':
            phase_correction = (bdelta-Delta_S_Sampling)*((2*np.pi)/bdelta)
        return phase_correction


    def single_prediction(self, y_signal, skip_msmts, init=[None, None],
                          basis_choice='A', prediction_method_default='ZeroGain'):
        ''' Return predictions for LKFFB. Prediction method default is ZeroGain.

        Parameters:
        ----------
            y_signal (`float64`) :  Noisy measurement data (input to filtering).
            skip_msmts (`int`): Number of time-steps to skip between measurements.
                To receive measurement at every time-step, set skip_msmts=1.
            init (`float64`, optional): List containing Kalman noise variances
                [sigma, R] for filtering. If None list, defaults to
                [Kalman.optimal_sigma, Kalman.optimal_R].
            basis_choice (`str`, optional): Choice of built-in basis ['A', 'B', 'C']
                set by a string character.
            prediction_method_default (`str`, optional): Choice of prediction
                method ['ZeroGain, 'PropForward']. Defaults to 'ZeroGain'.

        Returns:
        ------
            predictions : Returns Kalman predictions spanning state estimation
                and forecasting regions.
        '''

        if init[0] == None and init[1] == None:
            init = np.zeros(2)
            init[0] = self.optimal_sigma
            init[1] = self.optimal_R

        append_decriptor = self.filename_KF+'_fast_skipmsmts_'+str(skip_msmts)

        predictions = skf.kf_2017(y_signal, self.n_train, self.n_testbefore,
                                  self.n_predict, self.Delta_T_Sampling,
                                  self.x0, self.p0, init[0], init[1],
                                  self.basis_dict[basis_choice],
                                  phase_correction=self.phase_dict[basis_choice],
                                  prediction_method=prediction_method_default,
                                  skip_msmts=skip_msmts, descriptor=append_decriptor)
        return predictions

    def detailed_single_prediction(self, y_signal, skip_msmts, init=[None, None], basis_choice='A'):
        '''
        Return predictions and instantaneous amplitudes for LKFFB. Prediction method
        default is PropForward.

        Parameters:
        ----------
            y_signal (`float64`) :  Noisy measurement data (input to filtering).
            skip_msmts (`int`): Number of time-steps to skip between measurements.
                To receive measurement at every time-step, set skip_msmts=1.
            init (`float64`, optional): List containing Kalman noise variances
                [sigma, R] for filtering. If None list, defaults to
                [Kalman.optimal_sigma, Kalman.optimal_R].
            basis_choice (`str`, optional): Choice of built-in basis ['A', 'B', 'C']
                set by a string character.
            prediction_method_default (`str`, optional): Choice of prediction
                method ['ZeroGain, 'PropForward']. Defaults to 'ZeroGain'.

        Returns:
        ------
            predictions : Returns Kalman predictions spanning state estimation
                and forecasting regions.
            instantA : Returns amplitude information learned by LKFFB for each
                basis frequency.
        '''

        if init[0] == None and init[1] == None:
            init = np.zeros(2)
            init[0] = self.optimal_sigma
            init[1] = self.optimal_R

        append_decriptor = self.filename_KF+'_det_skipmsmts_'+str(skip_msmts)

        predictions, instantA = dkf.detailed_kf(append_decriptor, y_signal, self.n_train,
                                                self.n_testbefore,
                                                self.n_predict, self.Delta_T_Sampling,
                                                self.x0, self.p0, init[0], init[1],
                                                self.basis_dict[basis_choice],
                                                phase_correction=self.phase_dict[basis_choice],
                                                skip_msmts=skip_msmts)
        return predictions, instantA


    def ensemble_avg_predictions(self, skip_msmts, chosen_params=[None, None], NO_OF_KALMAN_VARIATIONS=6):
        '''
        Return ensemble averaged LKFFB prediction over Basis A,B,C and all Prediction
        Methods and save output of all runs as npz file.

        Parameters:
        ----------
            skip_msmts (`int`): Number of time-steps to skip between measurements.
                To receive measurement at every time-step, set skip_msmts=1.
            chosen_params (`float64`, optional): List containing Kalman noise variances
                [sigma, R] for filtering. If None list, defaults to
                [Kalman.optimal_sigma, Kalman.optimal_R] in Kalman.single_prediction().
            NO_OF_KALMAN_VARIATIONS (`int`) : Number of combinations of choice of
                LKFFB basis ['A', 'B', 'C'] and choice of prediction methods
                ['ZeroGain, 'PropForward']. Defaults to '6'.

        Returns:
        ------
            Saves .npz file with parameter regime and prediction analysis output.
        '''

        truth_datasets = np.zeros((self.number_of_points, self.max_it))
        KF_Predictions_Matrix = np.zeros((NO_OF_KALMAN_VARIATIONS, self.n_predict + self.n_testbefore, self.max_it))
        KF_Error_Means = np.zeros((NO_OF_KALMAN_VARIATIONS, self.n_predict+self.n_testbefore))
        Predict_Zero_Means = np.zeros((self.n_testbefore+self.n_predict))

        for run in xrange(self.max_it): # Loop over ensemble size

            truth, y_signal = self.generate_data_from_truth(self.user_defined_variance)
            truth_datasets[:, run] = truth
            Predict_Zero_Means += (1.0/float(self.max_it))*sqr_err(np.zeros(self.n_testbefore + self.n_predict ), truth[ self.n_train - self.n_testbefore : self.n_train +self.n_predict])

            choice_counter = 0 # choice_counter takes values from 0, 1, ..., NO_OF_KALMAN_VARIATIONS -1

            for choice1 in self.basis_list: # Loop over Basis A, B, C

                for choice2 in self.prediction_method_list: # Loop over Prediction Methods

                    predictions = self.single_prediction(y_signal, skip_msmts, init=chosen_params, basis_choice=choice1, prediction_method_default=choice2)
                    KF_Error_Means[choice_counter, :] += (1.0/float(self.max_it))*sqr_err(predictions, truth[self.n_train-self.n_testbefore : self.n_train + self.n_predict])
                    KF_Predictions_Matrix[choice_counter, :, run] = predictions

                    choice_counter += 1

        Normalised_Means = KF_Error_Means / Predict_Zero_Means

        filename_KF_skip = self.filename_KF+'_skipmsmts_'+str(skip_msmts)

        np.savez(os.path.join(self.savetopath, filename_KF_skip),
                 end_run=run, filname0=self.filename0,
                 max_it=self.max_it,
                 savetopath=self.savetopath,
                 expt_params=self.expt_params,
                 kalman_params=self.kalman_params,
                 optimal_sigma=self.optimal_sigma,
                 optimal_R=self.optimal_R,
                 chosen_params=chosen_params,
                 skip_msmts=skip_msmts,
                 msmt_noise_params=self.msmt_noise_params,
                 true_noise_params=self.true_noise_params,
                 true_signal_params=self.true_signal_params,
                 user_defined_variance=[self.user_defined_variance],
                 msmt_noise_variance=self.msmt_noise_variance,
                 truth_datasets=truth_datasets,
                 Predict_Zero_Means=Predict_Zero_Means,
                 KF_Predictions_Matrix=KF_Predictions_Matrix,
                 KF_Error_Means=KF_Error_Means,
                 Normalised_Means=Normalised_Means)


    def convert_amp_hz_to_radians(self, frequency_axis, fourier_amplitudes):
        ''' Return PSD in radians based on a frequency axis in Hz (x-data) and
        Fourier amplitudes (signal units). Assumes Fourier amplitudes**2 estimates
        one realisation of the spectrum for a covariance stationary random process.
        '''

        omega_axis = 2*np.pi*frequency_axis
        one_PSD_realisation = (fourier_amplitudes**2)*2*np.pi
        return omega_axis, one_PSD_realisation


    def run_test_KF(self, skip_msmts=1, savefig='Yes'):
        ''' Compare KF prediction output for the given parameters, using different LKFFB
        variants. [DEPRECIATED]
        '''
        # Create predictions
        truth, y_signal = self.generate_data_from_truth(self.user_defined_variance)
        pred_skf = self.single_prediction(y_signal, skip_msmts=skip_msmts)
        #pred_skf_2, amp_skf = self.single_prediction(y_signal)
        pred_dkf, amp_dkf = self.detailed_single_prediction(y_signal, skip_msmts=skip_msmts)

        # Generate true PSD linne
        self.beta_z_truePSD()

        # Generated estimates of PSD by squaring Kalman learned amplitudes
        # x_skf, y_skf = self.convert_amp_hz_to_radians(self.basisA, amp_skf)
        x_dkf, y_dkf = self.convert_amp_hz_to_radians(self.basisA, amp_dkf)
        # x_data = [x_skf, x_dkf, self.true_w_axis[self.J -1:]]
        x_data = [x_dkf, self.true_w_axis[self.J - 1:]]
        # y_data = [y_skf*FUDGE, y_dkf*FUDGE, HILBERT_TRANSFORM*self.true_S_twosided[self.J -1:]]
        y_data = [y_dkf*FUDGE, HILBERT_TRANSFORM*self.true_S_twosided[self.J - 1:]]
        # FUDGE: numerical analysis shows my calc is off by 1/2
        # HILBERT_TRANSFORM: Comparison with the Hilbert Transform (amplitudes) from KF means we double the twosided spectrum

        time_predictions = [pred_skf, pred_dkf, truth[self.n_train-self.n_testbefore:self.n_train + self.n_predict], y_signal[self.n_train-self.n_testbefore:self.n_train + self.n_predict] ]#, pred_skf_2]
        lbl_list = ['KF (Fast)', 'Detailed KF', 'Truth', 'Msmts', 'KF_2']
        color_list = ['purple', 'green', 'red', 'black', 'purple']
        markr_list = ['o', 'H', '-', 'x', 'x--']
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 10))

        ax = axes[0]

        for i in xrange(len(time_predictions)):
            ax.plot(self.Time_Axis[self.n_train-self.n_testbefore:self.n_train + self.n_predict],
                    time_predictions[i], markr_list[i], color=color_list[i],
                    alpha=0.5, label=lbl_list[i])

        ax.axhline(0.0, color='black', label='Predict Zero')
        ax.axvline(self.n_train*self.Delta_T_Sampling, linestyle='--', color='gray', label='Training Ends')

        ax.set(xlabel='Time [s]', ylabel="n-Step Ahead Msmt Prediction [Signal Units]")
        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand",
                  borderaxespad=0, frameon=False, fontsize=14)

        ax = axes[1]
        ax.set(xlabel='Omega [radians]')
        ax.set_ylabel(r'$A_{KF}^2$ vs. PSD [Power/radians]')

        num_amps = len(x_data) -1
        for i in xrange(num_amps):
            ax.plot(x_data[i], y_data[i], markr_list[i], alpha=0.5, markersize=8.0,
                    color=color_list[i],
                    label=lbl_list[i]+', Power: %s'%(np.round(np.sum(y_data[i]))))
        ax.plot(x_data[num_amps], y_data[num_amps], 'r', label=lbl_list[2]+', Power: %s'%(np.round(self.true_S_norm)))

        ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=1, mode="expand",
                  borderaxespad=0, frameon=False, fontsize=14)

        for ax in axes.flat:
            for item in (ax.get_xticklabels()):
                item.set_rotation('vertical')

            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(14)

        fig.subplots_adjust(left=0.0, right=0.99, wspace=0.2, hspace=0.2, top=0.8, bottom=0.2)
        fig.suptitle('Theoretical Truth v. Learned Kalman Predictions Using Basis A', fontsize=14)
        # plt.show()

        if savefig == 'Yes':
            figname = os.path.join(self.savetopath, self.filename0+'_KF_run_test.svg',)
            fig.savefig(figname, format="svg")

        plt.close(fig)
