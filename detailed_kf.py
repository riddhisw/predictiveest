#descriptor = "DetailedKF_"
import numpy as np
import math
from timeit import default_timer as timer
import matplotlib.pyplot as plt


def detailed_kf(descriptor, y_signal, n_train, n_testbefore, n_predict, Delta_T_Sampling, x_hat_initial,P_hat_initial, oekalman, rkalman, freq_basis_array, phase_correction):
    
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
    n_converge -- Predicted timestep at which algorithm is expected to finish learning [Scalar int]
    phase_correction -- Applies if y_signal data are Ramsey frequency offset measurements [Scalar float64]
    
    
    Returns: 
    --------
    Saves two Numpy.savez files containing all inputs, parameters, intermediary calculations and checks. Predictions stored as:
    Propagate_Foward -- Output predictions. Non-zero only for n_converge < timestep < num. [Len: num. dtype = float64]
    
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
    
    instantA_Prediction -- Value of instantaneous amplitude (instantA) at n_converge [Dim: numf x 1. dtype = float64] 
    instantP_Prediction -- Value of instantaneous phase (instantP) at n_converge [Dim: numf x 1. dtype = float64] 
    
    Dist_Prediction -- Difference between measurements and 'propagated' predictions for timesteps > n_converge [Dim: (num - n_converge) x 1. dtype = float64] 
    
    '''
    
    phase_correction_noisetraces = phase_correction
    n_converge = n_train

    # Model Dimensions
    num = n_predict + n_train
    numf = len(freq_basis_array)
    twonumf = numf*2

    # Kalman Measurement Data
    z = np.zeros((1,1,num)) 
    z[0,0,:] = y_signal
    
    # State Estimation
    x_hat = np.zeros((twonumf,1,num)) 
    z_proj = np.zeros((1,1,num)) 
    e_z = np.zeros((1,1,num)) 
    
    # Noise Features
    Gamma2 = np.zeros((twonumf,1,num)) 
    Gamma = np.zeros((twonumf,1,num)) 
    Q = np.zeros((twonumf,twonumf,num)) 
    R = np.ones((1,1,num)) 
    R[0,0,:] = rkalman
    
    # Covariance Estimation
    S = np.zeros((1,1,num))
    S_inv = np.zeros((1,1,num)) 
    W = np.zeros((twonumf,1,num)) 
    P_hat = np.zeros((twonumf,twonumf,num)) 
    
    # Dynamical Model, Measurement Action and Initial Conditions
    a = np.zeros((twonumf,twonumf)) 
    h = np.zeros((1,twonumf,num)) 
    h[0,0,:] = 1.0
    
    coswave = -1 # -1 for a cosine state, +1 for a sine state signal. -1 will work for a sine wave with random phases.
    comp = 0
    for comp in range(0,twonumf,2):
        a[comp,comp] = np.cos(Delta_T_Sampling*freq_basis_array[comp/2]*2*np.pi)
        a[comp+1,comp+1] =  np.cos(Delta_T_Sampling*freq_basis_array[comp/2]*2*np.pi)
        a[comp,comp+1] = coswave*np.sin(Delta_T_Sampling*freq_basis_array[comp/2]*2*np.pi)
        a[comp+1,comp] = -a[comp,comp+1] 
        
    initial = 0
    for initial in range(0,twonumf,1):
        P_hat[initial,initial,0] = P_hat_initial 
        x_hat[initial,0,0] = x_hat_initial 
        if (initial % 2 == 0):
            h[0,initial,:] = 1.0

    # Instantaneous Amplitude, Phase and Frequency Calculations
    instantA = np.zeros((numf,num)) 
    instantP = np.zeros((numf,num)) 
    instantW = np.zeros((numf,num)) 
    
    # print 'Initial State', x_hat
    # print 'Initial State', P_hat
    # print h
    # print a
    print(freq_basis_array)
    k = 1
    while (k< num):

        #print 'Apriori Predicted State x_hat'
        x_hat[:,:,k] = np.dot(a,x_hat[:,:,k-1]) #Predicted state prior to measurement (no controls) for no dynamic model and no process noise
        
        if k<10:
            print "DKF, timestep ", k
            print('a', a)
            print('h', h)
            print('xhat aprioir')
            print (x_hat[:,:,k])
        #print 'Evolve Process Noise Features' i.e. Gamma * Gamma T. Use k-1 since we add Q[k-1] with P[k-1]
        spectralresult0=0
        spectralresult=0
        for spectralresult0 in range(0,len(freq_basis_array),1):
            spectralresult = spectralresult0*2    
            Gamma2[spectralresult,0,k-1] = x_hat[spectralresult,0,k-1]*(np.sqrt(oekalman**2/ (x_hat[spectralresult,0,k-1]**2 + x_hat[spectralresult + 1,0,k-1]**2)))
            Gamma2[spectralresult+1,0,k-1] = x_hat[spectralresult + 1,0,k-1]*(np.sqrt(oekalman**2/ (x_hat[spectralresult,0,k-1]**2 + x_hat[spectralresult + 1,0,k-1]**2)))

        Gamma[:,0,k-1] = np.dot(a,Gamma2[:,0,k-1] )
        if k == 1000:
            np.savetxt('Gamma_dkf', Gamma[:,0,k-1])
            np.savetxt('x_dkf', x_hat[:,:,k])
            
        Q[:,:,k-1] = np.dot(Gamma[:,:,k-1], Gamma[:,:,k-1].T)

        #print 'Apriori Predicted State Variance'
        P_hat[:,:,k] = np.dot(np.dot(a,P_hat[:,:,k-1]),a.T) + Q[:,:,k-1]

        #print 'Apriori Predicted Measurement'
        z_proj[0,0,k] = np.dot(h[:,:,k],x_hat[:,:,k]) #Predicted state at time k (one step ahead from k-1) 
        
        #print 'Apriori Predicted Measurement Variance, S, and  Gain Calculation'
        S[:,:,k] = np.dot(np.dot(h[:,:,k],P_hat[:,:,k]),h[:,:,k].T) + R[:,:,k] 
        S_inv[:,:,k] = np.linalg.inv(S[:,:,k])
        
        W[:,:,k] = np.dot(P_hat[:,:,k],h[:,:,k].T)*S_inv[:,:,k] 

        #print 'Measurement Residual'
        e_z[0,0,k] = z[0,0,k]-z_proj[0,0,k] 

        #print 'Aposteriori Updates'
        x_hat[:,:,k] = x_hat[:,:,k] + W[:,:,k]*e_z[0,0,k] 
        P_hat[:,:,k] = P_hat[:,:,k] - S[:,:,k]*np.dot(W[:,:,k],W[:,:,k].T) # For scalar S
        
        if k<10:
            print
            print
            print('S', S_inv[:,:,k])
            print('1/S', 1.0/S[:,:,k])
            print
            print('Gain')
            print(W[:,:,k])
            print
            print 
            print('Q')
            print(Q[:,:,k-1])
            print
        k=k+1
    
    ## CALCULATE INSTANTANEOUS PHASE, AMPLITUDE AND FREQUENCY
    k=1
    while (k< num):
        spectralresult0=0
        spectralresult=0
        for spectralresult0 in range(0,len(freq_basis_array),1):
            spectralresult = spectralresult0*2    
            instantA[spectralresult0,k] = np.sqrt(x_hat[spectralresult,0,k]**2 + x_hat[spectralresult + 1,0,k]**2) # using apostereroiri estimates
            instantP[spectralresult0,k] = math.atan2(x_hat[spectralresult + 1,0,k],x_hat[spectralresult,0,k]) # correct phase using atan2
            instantW[spectralresult0,k] = (1/(2*np.pi))*(((x_hat[spectralresult,0,k])*(x_hat[spectralresult + 1,0,k]-x_hat[spectralresult + 1,0,k-1])-x_hat[spectralresult + 1,0,k]*(x_hat[spectralresult,0,k]-x_hat[spectralresult,0,k-1]))/Delta_T_Sampling)/(x_hat[spectralresult,0,k]**2 + x_hat[spectralresult + 1,0,k]**2)
        k=k+1

    #print 'Calculating reconstructed state...'

    ## CALCULATE RECONSTRUCTED STATE
    spectralresult=0
    reconstructstate = 0
    for spectralresult in range(0,len(freq_basis_array),1):
        reconstructstate += x_hat[spectralresult*2,0,:] + 1j*x_hat[spectralresult*2 + 1,0,:]

    #print 'Calculating propagation forward....'

    ## PROPAGATE FORWARD USING HARMONIC SUMS
    print("PHASE CORR KF DKF", phase_correction_noisetraces)
    np.savetxt("IA_0_DKF", instantA[0,:])

    Propagate_Foward = np.zeros((num))
    instantA_Prediction = instantA[:,n_converge]
    instantP_Prediction = instantP[:,n_converge]

    #Using a harmonic sum for the noise 
    tn = 0
    Harmonic_Component = 0.0
    for tn in range(n_converge,num,1):
        for spectralcomponent in range(0, len(freq_basis_array)):
            if freq_basis_array[spectralcomponent] == 0:
                Harmonic_Component = instantA_Prediction[spectralcomponent]*math.cos((Delta_T_Sampling*tn*freq_basis_array[spectralcomponent]*2*np.pi + instantP_Prediction[spectralcomponent]))
            if freq_basis_array[spectralcomponent] != 0:
                Harmonic_Component = instantA_Prediction[spectralcomponent]*math.cos((Delta_T_Sampling*tn*freq_basis_array[spectralcomponent]*2*np.pi + instantP_Prediction[spectralcomponent] + phase_correction_noisetraces)) # with correction for noise traces 
            Propagate_Foward[tn] += Harmonic_Component


    np.savez(str(descriptor),descriptor=descriptor, y_signal=y_signal,instantP=instantP,freq_basis_array= freq_basis_array, instantA=instantA,Propagate_Foward=Propagate_Foward, x_hat=x_hat, P_hat=P_hat, W=W, Q=Q)
    
    return Propagate_Foward[n_converge:], instantA_Prediction
