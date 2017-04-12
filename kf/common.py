from __future__ import division, print_function, absolute_import

import numpy as np
import numba as nb
import numpy.linalg as la

@nb.jit(nopython=True)
def calc_inst_params(x_hat_time_slice):
    '''
    Returns instantaneous amplitudes and instaneous phases associated with each Kalman basis osccilator using state estimate, x_hat, at a given time step. 
    '''
    instantA_slice = np.sqrt(x_hat_time_slice[::2,0]**2 + x_hat_time_slice[1::2, 0]**2) # using apostereroiri estimates
    instantP_slice = np.arctan2(x_hat_time_slice[1::2,0], x_hat_time_slice[::2,0]) # correct phase using atan2
    
    # Changed math.atan2 to numpy.atan2 to support vectoristion.
    return instantA_slice, instantP_slice


@nb.jit(nopython=True)
def calc_pred(x_hat_series):
    
    '''
    Keyword Arguments:
    ------------------
    x_hat_series -- Aposteriori estimates (real and estimated imaginary components of the state for each basis frequency) for num_of_time_steps [Dim: twonumf x num_of_time_steps. dtype = float64]
    
    Returns:
    ------------------
    pred -- Measurement predictions based on adding the real parts of x_hat [Len: twonumf. dtype = float64]
    '''
    
    series = x_hat_series.shape[2]
    pred = np.zeros(series)
    for k in xrange(series):
        pred[k] = np.sum(x_hat_series[::2, 0, k])
    return pred


@nb.jit(nopython=True)
def calc_Gamma(x_hat, oe, numf):
    '''Returns a vector of noise features used to calculate Q in Kalman Filtering
    '''
    Gamma2 = np.zeros((2*numf,1))
    spectralresult0=0
    spectralresult=0
    for spectralresult0 in xrange(numf):
        spectralresult = spectralresult0*2
        Gamma2[spectralresult,0] = x_hat[spectralresult,0]*(np.sqrt(oe**2/ (x_hat[spectralresult,0]**2 + x_hat[spectralresult + 1,0]**2)))
        Gamma2[spectralresult+1,0] = x_hat[spectralresult + 1,0]*(np.sqrt(oe**2/ (x_hat[spectralresult,0]**2 + x_hat[spectralresult + 1,0]**2)))   
    return Gamma2