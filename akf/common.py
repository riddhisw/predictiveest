'''
Created on Thu Apr 20 19:20:43 2017
@author: riddhisw

.. module:: akf.common

    :synopsis: Simplies the AKF optimisation problem by using autoregressive weight
        estimates from the LSF.

    Module Level Functions:
    ----------------------
        fetch_weights : Return LSF weights to approximate the dynamical model in AKF.

.. moduleauthor:: Riddhi Gupta <riddhi.sw@gmail.com>

'''

from __future__ import division, print_function, absolute_import
import numpy as np

def fetch_weights(dataobject, choose_stp_fwds=1):
    ''' Return LSF weights to approximate the dynamical model in AKF.

    Parameters:
    ----------
        dataobject (`obj`):   instance.
        choose_stp_fwds (`int`):  Extract learned coefficients from LSF output.
            Defaults to 1 [Dim: 1x1].

    Returns:
    -------
        chosen_weights (`float64`):  AR(q) coefficient estimates based on LSF output [Dim: 1 x q].
    '''

    LSF_mean_weights = np.mean(dataobject.LSF_macro_weights[:,:,:,0],  axis=0)

    # choose_stp_fwds = 1 implies we choose the best model,
    # i.e. the one step ahead weights in AR(q) LSF

    chosen_weights = LSF_mean_weights[choose_stp_fwds, :]
    return chosen_weights

def ar_coefficients(arma_eigvals):
    ''' Return coefficients, {c_i}, of AR(p) process such that dy (t+v)/dw(t) =  \sum_{i=1}^{p} c_{i} \lambda_{i}^v, for lag v.
    Where \lambda_i are eigen values of the matrix returned by function arma.kf.get_autoreg_model

    [Depreciated : Helper function for AR analysis (not used in Kalman Filtering)]
    '''

    order = arma_eigvals.shape[0]
    coeff = np.zeros(order, dtype=complex)

    for idx_k in xrange(order):
        product = np.prod(arma_eigvals[idx_k] - np.delete(arma_eigvals, (idx_k))) 
        coeff[idx_k] = arma_eigvals[idx_k]**(order-1) / product

    return coeff


def ar_covariancefunc(akf_eigval, akf_coeff, max_lag=2100):
    ''' Return dy (t+v)/dw(t) for different lags, v, the covariance function of an AR(p) process such that dy (t+v)/dw(t) =  \sum_{i=1}^{p} c_{i} \lambda_{i}^v
    Where \lambda_i are eigen values of the matrix returned by function arma.kf.get_autoreg_model

    [Depreciated : Helper function for AR analysis (not used in Kalman Filtering)]
    '''

    covariance_func= np.zeros(max_lag, dtype=complex)
    for idx_lag in xrange(max_lag):
        covariance_func[idx_lag] = np.dot(akf_eigval**idx_lag, akf_coeff)

    return covariance_func
