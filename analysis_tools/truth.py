#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Created on Thu Apr 20 19:20:43 2017
@author: riddhisw

.. module:: analysis_tools.truth

    :synopsis: Defines true stochastic state and its properties; as dictated by
                (test_case, variation). (Algorithms are blind to these properties.)

    Module Level Classes:
    ----------------------
        Truth : Defines true dephasing field and its properties. The dephasing
            field is used to generate simulated experimental datasets. Learning
            algorithms are blind to the the properties of the dephasing field.

.. moduleauthor:: Riddhi Gupta <riddhi.sw@gmail.com>
"""

from __future__ import division, print_function, absolute_import
import numpy as np

Moments = {'Mean':np.mean, 'Variance':np.var}


# SCALING FACTOR FOR ALL LKKFB AMPLITUDES
# LKFFB acts on Hilbert transform of the true signal.
# Negative frequencies are shifted to postive spectrum.
# For a real, covariance stationary signal, the spectrum is symmetric
# Hence, our application of LKKFB means that we will estimate twice the true spectrum
LKFFB_HILBERT_TRANSFORM_ = 0.5

# SCALING FACTOR FOR ALL TRUTHS
# We convert a true, two sided power spectral density into a one sided power spectral density
# For a real, covariance stationary signal, the spectrum is symmetric
# int_(-w)^(w) S(w) dw = 2 int_(0)^(w) S(w) dw
# So if we take only the postive end of the LHS, we have to multiply it by 2.0 to conserve power.
SYMMETRIC_ONE_SIDED_PSD = 2.0

# OTHER SCALING FACTORS
NUM_SCALE = 2.0*np.pi # Numerical averaging vs. Theory # not required for paper

class Truth(object):
    ''' Defines true dephasing field and its properties. The dephasing field is
        used to generate simulated experimental datasets. Learning algorithms
        are blind to the the properties of the dephasing field.

    Attributes:
    ----------
        true_noise_params (`float64`) : Parametes to initiate Truth class in order:
            apriori_f_mean (`float64`) : Dephasing noise mean.
            pdf_type (`str`) : 'Uniform' or 'Normal' - specifies the distribution from
                which random phases are drawn.
            alpha  (`float64`) : True noise global noise scale factor.
            f0 (`float64`) : True noise frequency spacing between adajcent components.
            p (`int`) :  True noise built-in choice of PSD for random phase noise field.
                            p = -2 <=> 1/f^2 dephasing noise field
                            p = -1 <=> 1/f dephasing noise field
                            p = 0 <=> 'white/flat' dephasing noise field
                            p = 1 <=> ohmic dephasing noise field
            J (`int`) :  True noise parameter for setting the number of spectral components.
            jstart (`int`) : True noise parameter for setting the number of spectral components.
        num | number_of_points (`int`) :  Experimentally controlled total number
            of points.
        DeltaT | Delta_T_Sampling (`float`): Experimentally controlled sampling time.
        ensemble_size (`int`) : Number of runs used for a PSD reconstruction.
        num_w_axis (`float64`) : Frequency axis (radians) for numerical reconstruction
            of true noise PSD. Calculated by Truth.average_PSD().
        num_S_estimate (`float64`) : Numerical two sided power spectral density.
            Calculated by Truth.average_PSD().
        num_S_norm (`float64`) : Numerical total energy (norm) for two sided, bandlimited PSD.
            Calculated by Truth.average_PSD().
        true_w_axis (`float64`) : Frequency axis (radians) for theoretical true noise PSD.
            Calculated by Truth.beta_z_truePSD().
        true_S_twosided (`float64`) : Theoretical two sided power spectral density.
            Calculated by Truth.beta_z_truePSD().
        true_S_norm (`float64`) : theoretical total energy (norm) for two sided, bandlimited PSD.
            Calculated by Truth.beta_z_truePSD().

    Methods:
    -------
        beta_z : Return true stochastic state in discrete time, namely f(n).
        beta_z_truePSD : Calculate theoretical PSD for beta_z
        average_PSD : Calculate numerical PSD estimate using beta_z() realisations.
        norm_squared : Return magnitude squared of a vector
    '''

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

    def beta_z(self):

        '''Return true stochastic state in discrete time, namely f(n), as a periodic
        noise signal with random phase information.'''

        twopi = 2.0*np.pi
        list_of_t = np.linspace(0, self.number_of_points-1, self.number_of_points)
        list_of_j = np.arange(self.jstart, self.J, 1) # Define with J, j_start = 1
        J_ = self.J - 1 # Define dimensions of tensor sums using J_

        theta = np.random.uniform(low=0.0, high=2.0*np.pi, size=J_)
        freqtensor_tj = np.cos((twopi*self.f0*self.Delta_T_Sampling*list_of_j*np.ones((J_, self.number_of_points)).T).T * list_of_t + (theta*np.ones((self.number_of_points,J_))).T)
        amplitudes = map(lambda x: x*(x**(0.5*self.p - 1)), list_of_j)

        return self.alpha*twopi*self.f0*np.sum((amplitudes*freqtensor_tj.T).T, axis=0), # add comma to retain compatability with beta_z


    def beta_z_truePSD(self):
        '''Returns theoretical PSD for dephasing field.'''

        self.true_w_axis = 2.0*np.pi*self.f0*np.linspace(-self.J+1, self.J-1, 2*self.J-1)

        S_onesided = np.zeros(self.J)

        for j in range(self.jstart,self.J,1): # should start at jstart
            S_onesided[j] = 0.5*np.pi*(self.alpha**2)*((2*np.pi*self.f0)**2)*(j*(j**(0.5*self.p -1)))**2

        self.true_S_twosided = np.zeros(2*self.J-1)
        self.true_S_twosided[self.J-1:] = S_onesided # this is not over-writing at J-1 (checked for odd and even J)
        self.true_S_twosided[0:self.J] = S_onesided[::-1] #This overwrites true_S_twosided at J =0 but it doesn't matter

        self.true_S_norm = np.sum(self.true_S_twosided)

        pass # self.true_w_axis, self.true_S_twosided, self.true_S_norm


    def average_PSD(self):
        '''Returns PSD estimate for an ensemble of time domain realisations of
        dephasing field by performing both ensemble and time averaging.
        '''

        self.num_norms = np.zeros([self.ensemble_size,2])

        PSD_ensemble = np.zeros(self.number_of_points, dtype='complex128')
        for ensemble in xrange(self.ensemble_size):
            noise_realisation = self.beta_z()[0] # Note that beta_z returns noise trace and phases. The [0] index picks out the noise traces
            PSDnoise_realisation = np.abs((1.0/np.sqrt(self.number_of_points))*np.fft.fft(noise_realisation))**2 # Using an scaled FFT
            self.num_norms[ensemble,0] = self.norm_squared(noise_realisation) # Energy of the signal in the time domain
            self.num_norms[ensemble,1] = np.sum(PSDnoise_realisation) # Energy of the signal in the Fourier domain. These norms are equal
            PSD_ensemble += PSDnoise_realisation

        avg_PSD = (1.0/self.ensemble_size)*PSD_ensemble # Ensemble averaging
        self.num_S_estimate = NUM_SCALE*(1.0/self.number_of_points)*(avg_PSD) # Taking the limit with respect to time.

        #Total Power
        self.num_S_norm = np.sum(self.num_S_estimate) 
        # Difference in num_S_norm and one member of norms[:,1] arises from taking the limit over time. 
        # Difference in height of num_S_norm and true_S_norm at primary signal frequency is because small, non zero energy exists at other frequencies 
        self.num_w_axis = 2.0*np.pi*np.fft.fftfreq(self.number_of_points,d=self.Delta_T_Sampling)

        pass  #self.num_w_axis, self.num_S_estimate, self.num_norms, self.num_S_norm


    def norm_squared(self, signal):
        '''Returns the magnitude squared of a vector.'''
        return np.sum(np.abs(signal)**2)