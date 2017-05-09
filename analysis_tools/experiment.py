from __future__ import division, print_function, absolute_import
import numpy as np


class Experiment(object):
   
    def __init__(self, expt_params):

        # Experimental params
        self.expt_params = expt_params
        self.n_train = int(expt_params[0])
        self.n_predict = int(expt_params[1])
        self.n_testbefore = int(expt_params[2])
        self.multiplier = expt_params[3]
        self.bandwidth = expt_params[4]
        
        self.number_of_points, self.fs, self.Delta_S_Sampling, self.Delta_T_Sampling, self.Frequency_Axis, self.Time_Axis = self.experiment_params()
        
        pass


    def experiment_params(self):
        ''' Creates consistent time and frequency domain grid parameters.
        
        Keyword Arguments:
        ------------------
        n_train -- Number of training points
        n_predict -- Number of testing points outside the zone of current data 
        bandwidth (B) --  Assumed bandwidth of the true signal [Scalar float64]
        multiplier (r) --  Nquist multiplier, r>2 for no aliasing [Scalar float64]
        
        Returns:
        -------
        number_of_points -- Total number of points for simulation i.e training points + test points [Scalar int]
        Delta_S_Sampling -- Frequency domain spacing
        Delta_T_Sampling -- Time domain spacing (1/fs) (time between measurements)
        fs -- rB, Sampling frequency
        Frequency_Axis -- Frequency bins generated in discrete Fourier space
        Time_Axis -- Time axis
        
        '''
            
        self.number_of_points = self.n_predict + self.n_train
        self.fs = self.bandwidth*self.multiplier
        self.Delta_S_Sampling = self.fs/self.number_of_points
        self.Delta_T_Sampling = 1.0/self.fs
    
        self.Frequency_Axis = np.fft.fftfreq(self.number_of_points, d=self.Delta_T_Sampling)
    
        self.Time_Axis = np.zeros(self.number_of_points)
        taxis = 0
        for taxis in range(0, self.number_of_points,1):
            self.Time_Axis[taxis] = taxis*self.Delta_T_Sampling
        
        return self.number_of_points, self.fs, self.Delta_S_Sampling, self.Delta_T_Sampling, self.Frequency_Axis, self.Time_Axis

