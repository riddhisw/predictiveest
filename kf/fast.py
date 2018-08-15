'''
.. module:: kf.fast

    :synopsis: LKFFB implementation with memoryless Kalman filtering. Facilitates exploration of different
        Prediction methods ('ZeroGain' or 'PropForward') and maximallly generic choices
        of apriori basis for LKFFB (Basis 'A', 'B', or 'C').

    Module Level Functions:
    ----------------------
        makePropForward : Return learned parameters from
            msmt_record via LKFFB and make predictions for timesteps > n_train.
        detailed_kf : Return LKFFB predictions.

.. moduleauthor:: Riddhi Gupta <riddhi.sw@gmail.com>

'''
import numpy as np
#import numba as nb
import numpy.linalg as la

from kf.common import (
    calc_inst_params, calc_pred, calc_Gamma, get_dynamic_model,
    propagate_states, calc_Kalman_Gain, calc_residuals
)

#@nb.jit(nopython=True) 
def makePropForward(freq_basis_array, x_hat, Delta_T_Sampling, phase_correction_noisetraces, num, n_train, numf):
    ''' Extracts learned parameters from Kalman Filtering msmt_record and makes
        predictions for timesteps > n_train.

    Parameters:
    ----------
        freq_basis_array (`float64`): Array containing `numf` number of basis frequencies.
        x_hat (`float64`): Aposteriori KF estimates based on msmt_record.
        Delta_T_Sampling (`float64`): Time interval between measurements.
        phase_correction_noisetraces (`float64`): Applies depending on choice of
            basis and prediction method.
        num (`int`): Number of points in msmt_record.
        n_train (`int`): Predicted timestep at which algorithm is expected to finish learning.
        numf (`int`): Number of points (spectral basis frequencies) in freq_basis_array.

    Returns:
    -------
        Propagate_Foward (`float64`): Output predictions. Non-zero only
            for n_train < timestep < num.
        instantA (`float64`): Instantaneous amplitude calculated based on
            estimated state x_hat [Dim: numf x num]
        instantP (`float64`): Instantaneous phase calculated based on estimated
            state x_hat [Dim: numf x num]
    '''
    instantA, instantP = calc_inst_params(x_hat)

    ## PROPAGATE FORWARD USING HARMONIC SUMS
    Propagate_Foward = np.zeros((num))

    tn = 0
    for tn in range(n_train, num, 1):
        Propagate_Foward[tn] = instantA[0]*np.cos((Delta_T_Sampling*tn*freq_basis_array[0]*2*np.pi + instantP[0]))
        Propagate_Foward[tn] += np.sum(instantA[1:]*np.cos((Delta_T_Sampling*tn*freq_basis_array[1:]*2*np.pi + instantP[1:] + phase_correction_noisetraces))) # with correction for noise traces 

    return Propagate_Foward, instantA, instantP


ZERO_GAIN, PROP_FORWARD = range(2)
PredictionMethod = {
    "ZeroGain": ZERO_GAIN,
    "PropForward": PROP_FORWARD
}

def kf_2017(y_signal, n_train, n_testbefore, n_predict, Delta_T_Sampling, x0, p0, oe, 
            rk, freq_basis_array, phase_correction=0 ,prediction_method="ZeroGain", 
            skip_msmts=1, descriptor='Fast_KF_Results'):
    ''' Return LKFFB predictions and save LKFFB analysis as .npz file.

    Parameters:
    ----------
    y_signal (`float64`): Array containing measurements for Kalman Filtering [Dim: 1 x num].
    n_train (`int`): Timestep at which algorithm is expected to finish learning.
    n_testbefore (`int`):  Number of on step ahead predictions prior to n_train
        which user requires to be returned as output.
    n_predict (`int`): Predictions outside of msmt data.
    Delta_T_Sampling (`float64`): Time interval between measurements.
    x0 (`float64`): x_hat_initial : Initial condition for state estimate, x(0), for all basis
        frequencies.
    p0 (`float64`): P_hat_initial : Initial condition for state covariance estimate, P(0),
        for all basis frequencies.
    oe (`float64`): oekalman : Process noise covariance strength.
    rk (`float64`): rkalman : Measurement noise covariance strength.
    freq_basis_array (`float64`): Array containing basis frequencies.
    phase_correction (`float64`): Basis dependent + prediction method dependent.
    prediction_method : Use ZeroGain OR PropagateForward with Phase Correction.
    skip_msmts : Allow a non zero Kalman gain for every n-th msmt,
            where skip_msmts == n and skip_msmts=1 implies all measurements
            can have a non-zero gain.

    Known Information for Filter Design:
    -------------------------------------------------------
    a -- Linearised dynamic model - time invariant [Dim: twonumf x twonumf. dtype = float64]
    h -- Linear measurement action - time invariant [Dim: 1 x twonumf. dtype = float64]
    Gamma2, Gamma -- Process noise features [Dim: twonumf x 1. dtype = float64]
    Q -- Process noise covariance.[Dim: twonumf x twonumf. dtype = float64]
    R -- Measurement noise covariance; equivalent to rkalman for scalar measurement
        noise. [Scalar float64]

    Variables for State Estimation and State Covariance Estimation:
    ---------------------------------------------------------------
    x_hat -- Aposteriori estimates (real and estimated imaginary components
        of the state for each basis frequency)  [Len: twonumf. dtype = float64].
    x_hat_apriori -- Apriori estimates (real and estimated imaginary components
        of the state for each basis frequency) [Len: twonumf. dtype = float64].
    z_proj -- Apriori predicted measurement [Scalar float64]
    e_z -- Residuals, i.e. z - z_proj [Len: num float64]
    S --  Predicted output covariance estimate (i.e. uncertainty in  z_proj)
        [Scalar float64].
    S_inv -- Inverse of S (NB: R must be a positive definite if S is not Scalar)
        [Scalar float64].
    W -- Kalman gain [Dim: twonumf x 1. dtype = float64]
    P_hat -- Aposteriori state covariance estimate (i.e. aposteriori uncertainty
        in estimated x_hat) [Dim: twonumf x twonumf. dtype = float64]
    P_hat_apriori -- Apriori state covariance estimate (i.e. apriori uncertainty in
        estimated x_hat) [Dim: twonumf x twonumf. dtype = float64]

    Returns:
    --------
    predictions (`float64`): Output predictions [Len: n_testbefore + n_predict].
    InstantA (`float64`): Instantaneous amplitudes at n_train use for generating predictions
        using Prop Forward [len: numf].

    Dimensions:
    -----------
    num (`int`): Number of points in msmt_record.
    numf (`int`): Number of points (spectral basis frequencies) in freq_basis_array.
    twonumf (`int`): 2*numf. (NB: For each basis freq in freq_basis_array, estimators
        have a real and imaginary parts).

    '''
    return _kf_2017(y_signal, n_train, n_testbefore, n_predict, Delta_T_Sampling, x0, p0, oe, rk, freq_basis_array, phase_correction, PredictionMethod[prediction_method], skip_msmts, descriptor)


def _kf_2017(y_signal, n_train, n_testbefore, n_predict, Delta_T_Sampling, x0, p0, oe, rk, freq_basis_array, phase_correction, prediction_method_, skip_msmts, descriptor):
    ''' [Wrapper Function] See kf_2017 docstring for detailed definitions. '''

    num = n_train + n_predict
    numf = len(freq_basis_array)
    twonumf = int(numf*2.0)
    
    # Kalman Measurement Data
    z = np.zeros(num)
    z[:] = y_signal

    # State Estimation
    x_hat_apriori = np.zeros((twonumf,1)) 
    x_hat = np.zeros((twonumf,1))
    e_z = np.zeros(num)
    P_hat_apriori = np.zeros((twonumf,twonumf))    
    P_hat = np.zeros((twonumf,twonumf))

    # Dynamical Model
    a = get_dynamic_model(twonumf, Delta_T_Sampling, freq_basis_array, coswave=-1)
    
    # Measurement Action
    h = np.zeros((1,twonumf)) 
    h[0,::2] = 1.0
    
    # Initial Conditions
    x_hat[:,0] = x0 
    diag_indx = range(0,twonumf,1)
    P_hat[diag_indx, diag_indx] = p0
    
    store_x_hat = np.zeros((twonumf,1,num))
    #store_P_hat = np.zeros((twonumf,twonumf,num))
    store_x_hat[:,:,0] = x_hat
    #store_P_hat[:,:,0] = P_hat  
    
    #store_W = np.zeros((twonumf,1,num)) 
    #store_S_Outer_W = np.zeros((twonumf,twonumf,num))
    #store_Q = np.zeros((twonumf,twonumf,num))
    #store_S = np.zeros((1,1,num))
    predictions = np.zeros(n_testbefore + n_predict)
    
    # Start Filtering
    k = 1
    while (k< num): 
        
        x_hat_apriori, P_hat_apriori, dumpQ = propagate_states(a, x_hat, P_hat, oe, numf)
        
        if prediction_method_ == ZERO_GAIN and k> (n_train):
            # This loop is equivalent to setting the gain to zero 
            x_hat = x_hat_apriori
            store_x_hat[:,:,k] = x_hat
            P_hat = P_hat_apriori
            #store_P_hat[:,:,k] = P_hat
            k = k+1 
            continue 
        
        W, S = calc_Kalman_Gain(h, P_hat_apriori, rk)    
        #store_S[:,:, k] = S
        
        #Skip msmts        
        if k % skip_msmts !=0:
            W = np.zeros((twonumf, 1))
            
        e_z[k] = calc_residuals(h, x_hat_apriori, z[k])
        
        x_hat = x_hat_apriori + W*e_z[k]
        #store_S_Outer_W[:,:,k] = S*np.outer(W,W.T)
        P_hat = P_hat_apriori - S*np.outer(W,W.T) #Equivalent to outer(W, W)
        
        store_x_hat[:,:,k] = x_hat
        #store_P_hat[:,:,k] = P_hat         
        #store_W[:,:,k] = W

           
        if prediction_method_ == PROP_FORWARD and (k==n_train):
            # This loop initiates propagation forward at n_train
            Propagate_Foward, instantA, instantP = makePropForward(freq_basis_array, x_hat,Delta_T_Sampling,phase_correction,num,n_train,numf)
            # We use previous state estimates to "predict" for n < n_train
            predictions[0:n_testbefore] = calc_pred(store_x_hat[:,:,n_train-n_testbefore:n_train])
            # We use Prop Forward to "forecast" for n> n_train
            predictions[n_testbefore:] = Propagate_Foward[n_train:]
            
            return predictions
        
        k=k+1
        
    predictions = calc_pred(store_x_hat[:,:,n_train-n_testbefore:])
        
    return predictions

