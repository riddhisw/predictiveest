#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  8 11:32:27 2017

@author: riddhisw
"""
from __future__ import division, print_function, absolute_import
import numpy as np
#import scipy.stats as pdf

#PDF = {'Uniform':pdf.uniform,'Gamma': pdf.gamma, 'Normal': pdf.norm}
Moments = {'Mean':np.mean, 'Variance':np.var}


class Truth(object):

    
    def __init__(self, true_noise_params, num=None, DeltaT=None, ensemble_size=50):

        #Optional args
        if num != None:
            self.number_of_points = num
        if DeltaT != None:
            self.Delta_T_Sampling = DeltaT
        
        # Truth params
        self.true_noise_params = true_noise_params
        self.apriori_f_mean = float(true_noise_params[0])
        self.pdf_type = 'Uniform' #true_noise_params[1]
        self.alpha = float(true_noise_params[2])
        self.f0 = float(true_noise_params[3])
        self.p = float(true_noise_params[4])
        self.J = int(true_noise_params[5]) + 1
        self.jstart = int(true_noise_params[6])
        self.ensemble_size = ensemble_size

        self.num_w_axis = None
        self.num_S_estimate = None
        self.num_norms = None
        self.num_S_norm = None
        
        self.true_w_axis = None
        self.true_S_twosided = None
        self.true_S_norm = None

        self.true_signal_params = [self.pdf_type, self.number_of_points, self.Delta_T_Sampling, self.alpha, self.f0, self.p, self.J, self.jstart]
        #self.true_signal_params = [0.0, self.number_of_points, self.Delta_T_Sampling, self.alpha, self.f0, self.p, self.J, self.jstart]

        pass


    def beta_z(self):
        
        '''Returns the sum of cosines with random phases, with input parameters:
        
        pdf_type = example_params[0] -- Distribution from which random phases are drawn. Can be 'Uniform' or 'Normal'. [dtype = string]
        N = example_params[1] -- Experimentally controlled parameter - total number of points, N (number_of_points). [dtype = int]
        delT= example_params[2] -- Experimentally controlled parameter - sampling time, Delta_T_Sampling. [dtype = float64]
        alpha= example_params[3]-- True noise parameter - alpha - global noise scale factor. [dtype = float64]
        f0 = example_params[4]-- True noise parameter - f0 - frequency spacing between adajcent components in noise. [dtype = float64]
        p= example_params[5]-- True noise parameter - p - sets choice of PSD for random phase noise field (see [2]):
            p = -2 <=> 1/f^2 dephasing noise field
            p = -1 <=> 1/f dephasing noise field
            p = 0 <=> 'white/flat' dephasing noise field
            p = 1 <=> ohmic dephasing noise field
        J= example_params[6]-- True noise parameter - J - number of spectral components in true noise - jstart [dtype = int]
       
        '''
        
        twopi = 2.0*np.pi
        list_of_t = np.linspace(0, self.number_of_points-1, self.number_of_points)
        list_of_j = np.arange(self.jstart, self.J, 1) # Define with J, j_start = 1
        J_ = self.J - 1 # Define dimensions of tensor sums using J_

        theta = np.random.uniform(low=0.0, high=2.0*np.pi, size=J_)
        freqtensor_tj = np.cos((twopi*self.f0*self.Delta_T_Sampling*list_of_j*np.ones((J_, self.number_of_points)).T).T * list_of_t + (theta*np.ones((self.number_of_points,J_))).T)
        amplitudes = map(lambda x: x*(x**(0.5*self.p - 1)), list_of_j)

        return self.alpha*twopi*self.f0*np.sum((amplitudes*freqtensor_tj.T).T, axis=0), # add comma to retain compatability with beta_z


    def beta_z_truePSD(self):
        '''Returns theoretical PSD for beta_z'''

        self.true_w_axis = 2.0*np.pi*self.f0*np.linspace(-self.J+1, self.J-1, 2*self.J-1)

        S_onesided = np.zeros(self.J)

        for j in range(self.jstart,self.J,1): # should start at jstart
            S_onesided[j] = 0.5*np.pi*(self.alpha**2)*((2*np.pi*self.f0)**2)*(j*(j**(0.5*self.p -1)))**2

        self.true_S_twosided = np.zeros(2*self.J-1)
        self.true_S_twosided[self.J-1:] = S_onesided # this is not over-writing at J-1 (checked for odd and even J)
        self.true_S_twosided[0:self.J] = S_onesided[::-1] #This overwrites true_S_twosided at J =0 but it doesn't matter

        self.true_S_norm = np.sum(self.true_S_twosided)

        pass #self.true_w_axis, self.true_S_twosided, self.true_S_norm


    def average_PSD(self):
        '''Returns PSD estimate for an ensemble of time domain beta_z signals 
        by performing both ensemble and time averaging'''

        self.num_norms = np.zeros([self.ensemble_size,2])

        PSD_ensemble = np.zeros(self.number_of_points, dtype='complex128')
        for ensemble in xrange(self.ensemble_size):
            noise_realisation = self.beta_z()[0] # Note that beta_z returns noise trace and phases. The [0] index picks out the noise traces
            PSDnoise_realisation = np.abs((1.0/np.sqrt(self.number_of_points))*np.fft.fft(noise_realisation))**2 # Using an scaled FFT
            self.num_norms[ensemble,0] = self.norm_squared(noise_realisation) # Energy of the signal in the time domain
            self.num_norms[ensemble,1] = np.sum(PSDnoise_realisation) # Energy of the signal in the Fourier domain. These norms are equal
            PSD_ensemble += PSDnoise_realisation

        fudge= 2.0*np.pi # Guessed numerically
        avg_PSD = (1.0/self.ensemble_size)*PSD_ensemble # Ensemble averaging
        self.num_S_estimate = fudge*(1.0/self.number_of_points)*(avg_PSD) #Taking the limit with respect to time.

        #Total Power
        self.num_S_norm = np.sum(self.num_S_estimate) 
        # Difference in num_S_norm and one member of norms[:,1] arises from taking the limit over time. 
        # Difference in height of num_S_norm and true_S_norm at primary signal frequency is because small, non zero energy exists at other frequencies 
        
        self.num_w_axis = 2.0*np.pi*np.fft.fftfreq(self.number_of_points,d=self.Delta_T_Sampling)
        
        pass  #self.num_w_axis, self.num_S_estimate, self.num_norms, self.num_S_norm


    def norm_squared(self, signal):
        '''Returns the magnitude squared of a vector'''
        return np.sum(np.abs(signal)**2)



#    def sum_noise(self):
#        '''
#        Returns unravelled random variables generated from beta_z for PSD analysis
#        '''
#        data = np.zeros([self.ensemble_size, self.true_signal_params[1]])
##        for en in xrange(self.ensemble_size):
 #           data[en,:] = self.beta_z(self.true_signal_params)[0]
 #       return np.ravel(data)
