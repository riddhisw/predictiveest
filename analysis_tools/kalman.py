
from __future__ import division, print_function, absolute_import

import os
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../") # look in the parent directory containing both kf and analysis_tools packages
from kf import fast as skf
from kf import detailed as dkf

# test
from analysis_tools.experiment import Experiment
from analysis_tools.noisydata import Noisy_Data

FUDGE = 0.5
HILBERT_TRANSFORM = 2.0

class Kalman(Experiment, Noisy_Data):

    def __init__(self, filename0, savetopath, max_it, expt_params, kalman_params, msmt_noise_params, true_noise_params, user_defined_variance=None):
        
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
        self.bdelta = kalman_params[4] #1.0/(self.Delta_T_Sampling*self.n_train)

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
        self.prediction_method_list =["ZeroGain", "PropForward"]

        pass


    def sqr_err(self, predictions, truth):
        ''' Returns the squared error sequence between predictions sequence and truth sequence
        '''
        return (predictions.real - truth.real)**2
    
    
    def calc_phase_correction(self, bdelta, Delta_S_Sampling, phase_correction):
        ''' Calculates phase correction for noise traces, else returns zero.
        '''      
        phase_correction_noisetraces = 0.0
        if phase_correction == 'Yes':
            phase_correction_noisetraces = (bdelta-Delta_S_Sampling)*((2*np.pi)/bdelta)
        return phase_correction_noisetraces


    def single_prediction(self, y_signal, skip_msmts, init=[None, None], basis_choice='A', prediction_method_default='ZeroGain'):
        '''
        Returns predictions based on data, parameters specified, choice of basis, prediction method.
        Prediction method default set to "ZeroGain"
        Returns skipped msmts if required.
        Under skf_amp, returns learned instaneous amplitudes for Prop Forward and None for ZeroGain.
        y_signal represents a sequence of noisy measurement outcomes.
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
        Returns predictions and inst. amplitudes based on data, parameters specified, choice of basis.
        Prediction method is always Prop Forward.
        Returns skipped msmts if required.
        y_signal represents a sequence of noisy measurement outcomes.
        '''
        
        if init[0] == None and init[1] == None:
            init = np.zeros(2)
            init[0] = self.optimal_sigma
            init[1] = self.optimal_R
            
        append_decriptor = self.filename_KF+'_det_skipmsmts_'+str(skip_msmts)
        
        predictions, instantA = dkf.detailed_kf(append_decriptor, y_signal, self.n_train, self.n_testbefore,
                                                self.n_predict, self.Delta_T_Sampling, 
                                                self.x0, self.p0, init[0], init[1], 
                                                self.basis_dict[basis_choice], 
                                                phase_correction=self.phase_dict[basis_choice],
                                                skip_msmts=skip_msmts)               
        return predictions, instantA


    def ensemble_avg_predictions(self, skip_msmts, chosen_params=[None, None], NO_OF_KALMAN_VARIATIONS=6):
        '''
        Returns ensemble averaged prediction data as npz file saved at location savetopath.
        '''
        
        truth_datasets = np.zeros((self.number_of_points, self.max_it))
        KF_Predictions_Matrix = np.zeros((NO_OF_KALMAN_VARIATIONS, self.n_predict+self.n_testbefore, self.max_it))
        KF_Error_Means = np.zeros((NO_OF_KALMAN_VARIATIONS, self.n_predict+self.n_testbefore))
        Predict_Zero_Means = np.zeros((self.n_testbefore+self.n_predict))

        for run in xrange(self.max_it): # Loop over ensemble size

            truth, y_signal = self.generate_data_from_truth(self.user_defined_variance)
            truth_datasets[:, run] = truth
            Predict_Zero_Means += (1.0/float(self.max_it))*self.sqr_err(np.zeros(self.n_testbefore + self.n_predict ), truth[ self.n_train - self.n_testbefore : self.n_train +self.n_predict])
            
            choice_counter = 0 # choice_counter takes values from 0, 1, ..., NO_OF_KALMAN_VARIATIONS -1
            
            for choice1 in self.basis_list: # Loop over Basis A, B, C
                
                for choice2 in self.prediction_method_list: # Loop over Prediction Methods
    
                    predictions = self.single_prediction(y_signal, skip_msmts, init=chosen_params, basis_choice=choice1, prediction_method_default=choice2)
                    KF_Error_Means[choice_counter, :] += (1.0/float(self.max_it))*self.sqr_err(predictions, truth[self.n_train-self.n_testbefore : self.n_train + self.n_predict])
                    KF_Predictions_Matrix[choice_counter, :, run] = predictions

                    choice_counter +=1

        Normalised_Means = KF_Error_Means/ Predict_Zero_Means
        
        filename_KF_skip = self.filename_KF+'_skipmsmts_'+str(skip_msmts)
        
        np.savez(os.path.join(self.savetopath, filename_KF_skip), 
                 end_run=run, filname0=self.filename0, max_it = self.max_it, 
                 savetopath=self.savetopath,
                 expt_params = self.expt_params, 
                 kalman_params= self.kalman_params, 
                 optimal_sigma = self.optimal_sigma,
                 optimal_R = self.optimal_R,
                 chosen_params=chosen_params,
                 skip_msmts = skip_msmts,
                 msmt_noise_params= self.msmt_noise_params, 
                 true_noise_params= self.true_noise_params, 
                 true_signal_params= self.true_signal_params,
                 user_defined_variance= [self.user_defined_variance],
                 msmt_noise_variance= self.msmt_noise_variance,
                 truth_datasets=truth_datasets,
                 Predict_Zero_Means=Predict_Zero_Means, 
                 KF_Predictions_Matrix=KF_Predictions_Matrix,
                 KF_Error_Means=KF_Error_Means, 
                 Normalised_Means=Normalised_Means)
        pass



    def convert_amp_hz_to_radians(self, frequency_axis, fourier_amplitudes):
        '''
        Returns PSD in radians based on a frequency axis in Hz (x-data) and Fourier amplitudes in signal units (signal units).
        Assumes Fourier amplitudes**2 == one realisation of a true PSD for a covariance stationary random process
        '''
        
        omega_axis = 2*np.pi*frequency_axis
        one_PSD_realisation = (fourier_amplitudes**2)*2*np.pi 
        return omega_axis, one_PSD_realisation


    def run_test_KF(self, skip_msmts=1, savefig='Yes'):
        '''
        Compares KF prediction output for the parameters given.
        '''
        # Create predictions
        truth, y_signal = self.generate_data_from_truth(self.user_defined_variance)
        pred_skf = self.single_prediction(y_signal, skip_msmts=skip_msmts)
        #pred_skf_2, amp_skf = self.single_prediction(y_signal)
        pred_dkf, amp_dkf = self.detailed_single_prediction(y_signal, skip_msmts=skip_msmts)

        # Generate true PSD linne
        self.beta_z_truePSD()

        # Generated estimates of PSD by squaring Kalman learned amplitudes       
        #x_skf, y_skf = self.convert_amp_hz_to_radians(self.basisA, amp_skf)
        x_dkf, y_dkf = self.convert_amp_hz_to_radians(self.basisA, amp_dkf)
        #x_data = [x_skf, x_dkf, self.true_w_axis[self.J -1:]]
        x_data = [ x_dkf, self.true_w_axis[self.J -1:]]
        #y_data = [y_skf*FUDGE, y_dkf*FUDGE, HILBERT_TRANSFORM*self.true_S_twosided[self.J -1:]]
        y_data = [y_dkf*FUDGE, HILBERT_TRANSFORM*self.true_S_twosided[self.J -1:]]
        # FUDGE: numerical analysis shows my calc is off by 1/2 
        # HILBERT_TRANSFORM: Comparison with the Hilbert Transform (amplitudes) from KF means we double the twosided spectrum

        
        time_predictions = [pred_skf, pred_dkf, truth[self.n_train-self.n_testbefore:self.n_train + self.n_predict], y_signal[self.n_train-self.n_testbefore:self.n_train + self.n_predict] ]#, pred_skf_2]
        lbl_list = ['KF (Fast)', 'Detailed KF', 'Truth', 'Msmts', 'KF_2']
        color_list = ['purple','green', 'red', 'black', 'purple']
        markr_list = ['o', 'H', '-', 'x', 'x--']
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,10))
        
        ax = axes[0]
        
        for i in xrange(len(time_predictions)):
            ax.plot(self.Time_Axis[self.n_train-self.n_testbefore:self.n_train + self.n_predict], 
                    time_predictions[i],markr_list[i], color = color_list[i], 
                    alpha=0.5,label=lbl_list[i])

        ax.axhline(0.0,  color='black',label='Predict Zero')
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
        plt.show()

        if savefig=='Yes':
            fig.savefig(self.filename0+'_run_test', format="svg")        

        pass
