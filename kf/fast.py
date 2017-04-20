import numpy as np
import numba as nb
import numpy.linalg as la

from kf.common import (
    calc_inst_params, calc_pred, calc_Gamma, get_dynamic_model,
    propagate_states, calc_Kalman_Gain, calc_residuals
)

#@nb.jit(nopython=True) 
def makePropForward(freq_basis_array, x_hat, Delta_T_Sampling, phase_correction_noisetraces, num, n_train, numf):
    ''' Extracts learned parameters from Kalman Filtering msmt_record and makes predictions for timesteps > n_train
    
    Keyword Arguments:
    ------------------
    freq_basis_array -- Array containing basis frequencies. [Len: numf. dtype = float64]
    x_hat -- Aposteriori KF estimates based on msmt_record (real and estimated imaginary components of the state for each basis frequency) [Len: twonumf. dtype = float64]
    Delta_T_Sampling -- Time interval between measurements. [Scalar int]
    phase_correction_noisetraces -- Applies depending on choice of basis [Scalar float64]
    num -- Number of points in msmt_record. [Scalar int]
    n_train -- Predicted timestep at which algorithm is expected to finish learning [Scalar int]
    numf -- Number of points (spectral basis frequencies) in freq_basis_array. [Scalar int]
    
    Returns: 
    --------
    Propagate_Foward -- Output predictions. Non-zero only for n_train < timestep < num. [Len: num. dtype = float64] 
    instantA -- Instantaneous amplitude calculated based on estimated state x_hat at n_train [Len: numf. dtype = float64] 
    instantP -- Instantaneous phase calculated based on estimated state x_hat at n_train [Len: numf. dtype = float64] 
       
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
    '''    
    Keyword Arguments:
    ------------------
    
    y_signal -- Array containing measurements for Kalman Filtering. [Dim: 1 x num. dtype = float64]
    n_train -- Timestep at which algorithm is expected to finish learning [Scalar int]
    n_testbefore --  Predictions in  zone of msmt data [Scalar int]
    n_predict -- Predictions outside of msmt data [Scalar int]
    Delta_T_Sampling -- Time interval between measurements. [Scalar int]
    x0 -- x_hat_initial -- Initial condition for state estimate, x(0), for all basis frequencies. [Scalar int]
    p0 -- P_hat_initial -- Initial condition for state covariance estimate, P(0), for all basis frequencies. [Scalar int]
    oe -- oekalman -- Process noise covariance strength. [Scalar int] 
    rk -- rkalman -- Measurement noise covariance strength. [Scalar int]
    freq_basis_array -- Array containing basis frequencies. [Len: numf. dtype = float64]
    phase_correction -- Basis dependent + prediction method dependent. [Scalar float64]
    prediction_method -- Basis dependent.  Use Use W=0 OR PropagateForward with Phase Correction.
    
    skip_msmts -- Allow a non zero Kalman gain for every n-th msmt, where skip_msmts == n
    
    Returns: 
    --------
    predictions -- Output predictions. [Len: n_testbefore + n_predict. dtype = float64]
    InstantA -- Instantaneous amplitudes at n_train use for generating predictions using Prop Forward [len: numf. dtype = float64]

    Dimensions:
    -----------
    num -- Number of points in msmt_record. [Scalar int]
    numf -- Number of points (spectral basis frequencies) in freq_basis_array. [Scalar int]
    twonumf -- 2*numf. (NB: For each basis freq in freq_basis_array, estimators have a real and imaginary parts). [Scalar int]
    
    Known Information for Filter Design:
    -------------------------------------------------------
    
    a -- Linearised dynamic model - time invariant [Dim: twonumf x twonumf. dtype = float64]
    h -- Linear measurement action - time invariant [Dim: 1 x twonumf. dtype = float64]
    Gamma2, Gamma -- Process noise features [Dim: twonumf x 1. dtype = float64]
    Q -- Process noise covariance.[Dim: twonumf x twonumf. dtype = float64]
    R -- Measurement noise covariance; equivalent to rkalman for scalar measurement noise. [Scalar float64]
    
    
    Variables for State Estimation and State Covariance Estimation:
    ---------------------------------------------------------------
    x_hat -- Aposteriori estimates (real and estimated imaginary components of the state for each basis frequency)  [Len: twonumf. dtype = float64]
    x_hat_apriori -- Apriori estimates (real and estimated imaginary components of the state for each basis frequency) [Len: twonumf. dtype = float64]
    z_proj -- Apriori predicted measurement [Scalar float64]
    e_z -- Residuals, i.e. z - z_proj [Len: num float64]
    
    S --  Predicted output covariance estimate (i.e. uncertainty in  z_proj) [Scalar float64] 
    S_inv -- Inverse of S (NB: R must be a positive definite if S is not Scalar) [Scalar float64]
    W -- Kalman gain [Dim: twonumf x 1. dtype = float64]
    P_hat -- Aposteriori state covariance estimate (i.e. aposteriori uncertainty in estimated x_hat) [Dim: twonumf x twonumf. dtype = float64]
    P_hat_apriori -- Apriori state covariance estimate (i.e. apriori uncertainty in estimated x_hat) [Dim: twonumf x twonumf. dtype = float64]
    
    '''    
    return _kf_2017(y_signal, n_train, n_testbefore, n_predict, Delta_T_Sampling, x0, p0, oe, rk, freq_basis_array, phase_correction, PredictionMethod[prediction_method], skip_msmts, descriptor)


def _kf_2017(y_signal, n_train, n_testbefore, n_predict, Delta_T_Sampling, x0, p0, oe, rk, freq_basis_array, phase_correction, prediction_method_, skip_msmts, descriptor):

    print(descriptor)
    print(prediction_method_)

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
    store_P_hat = np.zeros((twonumf,twonumf,num))
    store_x_hat[:,:,0] = x_hat
    store_P_hat[:,:,0] = P_hat  
    
    store_W = np.zeros((twonumf,1,num)) 
    store_S_Outer_W = np.zeros((twonumf,twonumf,num))
    store_Q = np.zeros((twonumf,twonumf,num))
    store_S = np.zeros((1,1,num))
    predictions = np.zeros(n_testbefore + n_predict)
    
    # Start Filtering
    k = 1
    while (k< num): 
        
        x_hat_apriori, P_hat_apriori, store_Q[:,:, k]= propagate_states(a, x_hat, P_hat, oe, numf)
        
        if prediction_method_ == ZERO_GAIN and k> (n_train):
            # This loop is equivalent to setting the gain to zero 
            x_hat = x_hat_apriori
            store_x_hat[:,:,k] = x_hat
            P_hat = P_hat_apriori
            store_P_hat[:,:,k] = P_hat
            k = k+1 
            continue 
        
        W, S = calc_Kalman_Gain(h, P_hat_apriori, rk)    
        store_S[:,:, k] = S
        
        #Skip msmts        
        if k % skip_msmts !=0:
            W = np.zeros((twonumf, 1))
            
        e_z[k] = calc_residuals(h, x_hat_apriori, z[k])
        
        x_hat = x_hat_apriori + W*e_z[k]
        store_S_Outer_W[:,:,k] = S*np.outer(W,W.T)
        P_hat = P_hat_apriori - S*np.outer(W,W.T) #Equivalent to outer(W, W)
        
        store_x_hat[:,:,k] = x_hat
        store_P_hat[:,:,k] = P_hat         
        store_W[:,:,k] = W

           
        if prediction_method_ == PROP_FORWARD and (k==n_train):
            # This loop initiates propagation forward at n_train
            Propagate_Foward, instantA, instantP = makePropForward(freq_basis_array, x_hat,Delta_T_Sampling,phase_correction,num,n_train,numf)
            # We use previous state estimates to "predict" for n < n_train
            predictions[0:n_testbefore] = calc_pred(store_x_hat[:,:,n_train-n_testbefore:n_train])
            # We use Prop Forward to "forecast" for n> n_train
            predictions[n_testbefore:] = Propagate_Foward[n_train:]
            
            np.savez(descriptor, 
                    descriptor=descriptor,
                    predictions=predictions, 
                    y_signal=y_signal,
                    freq_basis_array= freq_basis_array, 
                    x_hat=store_x_hat, 
                    P_hat=store_P_hat, 
                    a=a,
                    h=h,
                    z=z, 
                    e_z=e_z,
                    W=store_W,
                    Q=store_Q,
                    store_S_Outer_W=store_S_Outer_W,
                    S=store_S,
                    instantA=instantA,
                    instantP=instantP,
                    oe=oe, 
                    rk=rk,
                    n_train=n_train,
                    n_predict=n_predict,
                    n_testbefore=n_testbefore,
                    skip_msmts=skip_msmts,
                    Propagate_Foward=Propagate_Foward,
                    phase_correction=phase_correction)
            
            return predictions
        
        k=k+1
        
    predictions = calc_pred(store_x_hat[:,:,n_train-n_testbefore:])
    
    np.savez(descriptor, descriptor=descriptor,
             predictions=predictions, 
             y_signal=y_signal,
             freq_basis_array= freq_basis_array, 
             x_hat=store_x_hat, 
             P_hat=store_P_hat, 
             a=a,
             h=h,
             z=z,
             e_z=e_z,
             W=store_W,
             Q=store_Q,
             store_S_Outer_W=store_S_Outer_W,
             S=store_S,
             oe=oe, 
             rk=rk,
             n_train=n_train,
             n_predict=n_predict,
             n_testbefore=n_testbefore,
             skip_msmts=skip_msmts)
    
    return predictions

