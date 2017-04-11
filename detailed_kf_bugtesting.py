#descriptor = "DetailedKF_"
import numpy as np
import numba as nb

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


def detailed_kf(descriptor, y_signal, n_train, n_testbefore, n_predict, Delta_T_Sampling, x_hat_initial,P_hat_initial, oekalman, rkalman, freq_basis_array, phase_correction, skip_msmts=1):
    
    ''' Performs a full Kalman Filtering routine
    
    Keyword Arguments:
    ------------------
    
    y_signal -- Array containing measurements for Kalman Filtering. [Dim: 1 x 1 x num. dtype = float64]
    freq_basis_array -- Array containing basis frequencies. [Len: numf. dtype = float64]
    Delta_T_Sampling -- Time interval between measurements. [Scalar int]
    x_hat_initial -- Initial conditions for state estimate, x(0), for all basis frequencies. [Scalar int]
    P_hat_initial -- Initial conditions for state covariance estimate, P(0), for all basis frequencies. [Scalar int]
    oekalman -- Process noise covariance strength. [Scalar int] 
    rkalman -- Measurement noise covariance strength. [Scalar int]
    n_train /n_train -- Equivent to n_train if n_train is optimally chosen. Predicted timestep at which algorithm is expected to finish learning [Scalar int]
    phase_correction -- Applies if y_signal data are Ramsey frequency offset measurements [Scalar float64]
    skip_msmts -- Allow a non zero Kalman gain for every n-th msmt, where skip_msmts == n    
    n_testbefore -- Number of on step ahead predictions prior to n_train which user requires to be returned as output 
    
    Returns: 
    --------
    predictions -- Output predictions. Non-zero only for n_train < timestep < num. [Len: num. dtype = float64]
    InstantA -- Instantaneous amplitudes at n_train use for generating predictions using Prop Forward [len: numf. dtype = float64]
    
    Dimensions:
    -----------
    num -- Number of points in y_signal. [Scalar int]
    numf -- Number of points (spectral basis frequencies) in freq_basis_array. [Scalar int]
    twonumf -- 2*numf. (NB: For each basis freq in freq_basis_array, estimators have a real and imaginary parts). [Scalar int]
    
    
    Known Information for Filter Design:
    -------------------------------------------------------
    
    a -- Linearised dynamic model - time invariant [Dim: twonumf x twonumf. dtype = float64]
    h -- Linear measurement action - time invariant [Dim: 1 x twonumf x num. dtype = float64]
    Gamma2, Gamma -- Process noise features [Dim: twonumf x 1 x num. dtype = float64]
    Q -- Process noise covariance.[Dim: twonumf x twonumf x num. dtype = float64]
    R -- Measurement noise covariance; equivalent to rkalman for scalar measurement noise. [Dim: 1 x 1 x num. dtype = float64]
    
    
    Variables for State Estimation and State Covariance Estimation:
    ---------------------------------------------------------------
    x_hat -- Estimated real and estimated imaginary components of state [Dim: twonumf x 1 x num. dtype = float64]
    z_proj -- Apriori predicted measurement [Dim: 1 x 1 x num. dtype = float64]
    e_z -- Residuals, i.e. z - z_proj [Dim: 1 x 1 x num. dtype = float64]
    
    S --  Predicted output covariance estimate (i.e. uncertainty in  z_proj) [Dim: 1 x 1 x num. dtype = float64] 
    S_inv -- Inverse of S (NB: R must be a positive definite if S is not Scalar) [Dim: 1 x 1 x num. dtype = float64]
    W -- Kalman gain [Dim: twonumf x 1 x num. dtype = float64]
    P_hat -- State covariance estimate (i.e. uncertainty in estimated x_hat) [Dim: twonumf x twonumf x num. dtype = float64]
    
    
    Variables for Predictions (Propagating Forward):
    ------------------------------------------------
    instantA -- Instantaneous amplitude calculated based on estimated state x_hat [Dim: numf x num. dtype = float64] 
    instantP -- Instantaneous phase calculated based on estimated state x_hat [Dim: numf x num. dtype = float64] 
    instantW -- Instantaneous frequency calculated based on estimated state x_hat (not used) [Dim: numf x num. dtype = float64] 
    
    instantA_Prediction -- Value of instantaneous amplitude (instantA) at n_train [Dim: numf x 1. dtype = float64] 
    instantP_Prediction -- Value of instantaneous phase (instantP) at n_train [Dim: numf x 1. dtype = float64] 
    
    '''
    
    phase_correction_noisetraces = phase_correction
    predictions = np.zeros(n_testbefore + n_predict)

    # Model Dimensions
    num = n_predict + n_train
    numf = len(freq_basis_array)
    twonumf = int(numf*2)

    # Kalman Measurement Data
    z = np.zeros((1,1,num)) 
    z[0,0,:] = y_signal
    
    # State Estimation
    x_hat = np.zeros((twonumf,1,num)) 
    z_proj = np.zeros((1,1,num)) 
    e_z = np.zeros((1,1,num)) 
    
    # Noise Features
    Q = np.zeros((twonumf,twonumf,num)) 
    R = np.ones((1,1,num)) 
    R[0,0,:] = rkalman
    
    # Covariance Estimation
    S = np.zeros((1,1,num))
    S_inv = np.zeros((1,1,num)) 
    W = np.zeros((twonumf,1,num)) 
    P_hat = np.zeros((twonumf,twonumf,num)) 
    
    # Dynamical Model
    a = np.zeros((twonumf,twonumf))
    coswave = -1 
    index = range(0,twonumf,2)
    index2 = range(1,twonumf+1,2) #twnumf is even so need to add 1 to write over the last element
    diagonals = np.cos(Delta_T_Sampling*freq_basis_array*2*np.pi) #dim(freq_basis_array) = numf
    off_diagonals = coswave*np.sin(Delta_T_Sampling*freq_basis_array*2*np.pi)
    a[index, index] = diagonals
    a[index2, index2] = diagonals
    a[index, index2] = off_diagonals
    a[index2, index] = -1.0*off_diagonals
    
    # Measurement Action
    h = np.zeros((1,twonumf,num)) 
    h[0,::2,...] = 1.0
    
    # Initial Conditions
    x_hat[:,0,0] = x_hat_initial 
    diag_indx = range(0,twonumf,1)
    P_hat[diag_indx, diag_indx, 0] = P_hat_initial

    # Instantaneous Amplitude, Phase and Frequency Calculations # not optimised as it doesn't concern the loop
    instantA = np.zeros((numf,num)) 
    instantP = np.zeros((numf,num))
    
    k = 1
    while (k< num):

        #print 'Apriori Predicted State x_hat'
        x_hat[:,:,k] = np.dot(a,x_hat[:,:,k-1]) #Predicted state prior to measurement (no controls) for no dynamic model and no process noise
        
        #Removed code and added function calc_gamma 
        Gamma = np.dot(a,calc_Gamma(x_hat[:,:,k-1], oekalman, numf)) 
        Q[:,:,k-1] = np.dot(Gamma, Gamma.T)

        #print 'Apriori Predicted State Variance'
        P_hat[:,:,k] = np.dot(np.dot(a,P_hat[:,:,k-1]),a.T) + Q[:,:,k-1]

        #print 'Apriori Predicted Measurement'
        z_proj[0,0,k] = np.dot(h[:,:,k],x_hat[:,:,k]) #Predicted state at time k (one step ahead from k-1) 
        
        #print 'Apriori Predicted Measurement Variance, S, and  Gain Calculation'
        S[:,:,k] = np.dot(np.dot(h[:,:,k],P_hat[:,:,k]),h[:,:,k].T) + R[:,:,k] # implemented in detailed_kf.py
        S_comparison = np.dot(h[:,:,k], np.dot(P_hat[:,:,k],h[:,:,k].T)) + R[:,:,k] # should be implemented in detailed_kf to compare with multi_dot in kf_fast
        S_comparison2 = np.linalg.multi_dot([h[:,:,k], P_hat[:,:,k], h[:,:,k].T]) + R[:,:,k]  # multi_dot always returns return dot(A, dot(B, C)) since A==C.  implemented in memoryless KF
        
        #assert (np.linalg.norm(S[:, :, k] - S_comparison2, 2) <= 1e-11) # asserstion error not tripped so matrix multiplication dot(A, dot(B, C)) is the same as multi_dot
        S_inv[:,:,k] = 1.0/S[:,:,k]
        
        W[:,:,k] = np.dot(P_hat[:,:,k],h[:,:,k].T)*S_inv[:,:,k] 

        # Skp Msmts
        if k % skip_msmts !=0:
            W[:,:,k] = np.zeros_like(W[:,:,k]) #skipped msmt, model evolves with no new info.
        
        #print 'Measurement Residual'
        e_z[0,0,k] = z[0,0,k]-z_proj[0,0,k] 

        #print 'Aposteriori Updates'
        x_hat[:,:,k] = x_hat[:,:,k] + W[:,:,k]*e_z[0,0,k] 
        P_hat[:,:,k] = P_hat[:,:,k] - S[:,:,k]*np.dot(W[:,:,k],W[:,:,k].T) # For scalar S

        k=k+1
    
    ## CALCULATE INSTANTANEOUS PHASE, AMPLITUDE AND FREQUENCY    
    k=1
    while (k< num):
        instantA[:, k], instantP[:, k] = calc_inst_params(x_hat[:,:,k])
        k=k+1

    ## PROPAGATE FORWARD USING HARMONIC SUMS
    Propagate_Foward = np.zeros((num))
    instantA_Prediction = instantA[:,n_train]
    instantP_Prediction = instantP[:,n_train]

    #Using a harmonic sum for propagating  the noise 
    tn = 0
    for tn in range(n_train,num,1):
        Propagate_Foward[tn] = instantA_Prediction[0]*np.cos((Delta_T_Sampling*tn*freq_basis_array[0]*2*np.pi + instantP_Prediction[0]))
        Propagate_Foward[tn] += np.sum(instantA_Prediction[1:]*np.cos((Delta_T_Sampling*tn*freq_basis_array[1:]*2*np.pi + instantP_Prediction[1:] + phase_correction_noisetraces))) # with correction for noise traces 

    # We report one step ahead predictions for n < n_train
    predictions[0:n_testbefore] = calc_pred(x_hat[:,:,n_train-n_testbefore:n_train])
    # We use Prop Forward to "forecast" for n> n_train
    predictions[n_testbefore:] = Propagate_Foward[n_train:]

    np.savez(str(descriptor),
             descriptor=descriptor,
             predictions=predictions, 
             y_signal=y_signal,
             freq_basis_array= freq_basis_array, 
             x_hat=x_hat, 
             P_hat=P_hat, 
             a=a,
             h=h,
             z=z, 
             e_z=e_z,
             W=W, 
             Q=Q,
             instantA=instantA,
             instantP=instantP,
             n_train=n_train,
             Propagate_Foward=Propagate_Foward,
             phase_correction=phase_correction_noisetraces)

    return predictions, instantA_Prediction
