from __future__ import division, print_function, absolute_import

import numpy as np
#import numba as nb
import numpy.linalg as la


#@nb.jit(nopython=True)
def calc_inst_params(x_hat_time_slice):
    '''
    Returns instantaneous amplitudes and instaneous phases associated with each Kalman basis osccilator using state estimate, x_hat, at a given time step. 
    '''
    instantA_slice = np.sqrt(x_hat_time_slice[::2,0]**2 + x_hat_time_slice[1::2, 0]**2) # using apostereroiri estimates
    instantP_slice = np.arctan2(x_hat_time_slice[1::2,0], x_hat_time_slice[::2,0]) # correct phase using atan2
    
    # Changed math.atan2 to numpy.atan2 to support vectoristion.
    return instantA_slice, instantP_slice


#@nb.jit(nopython=True)
def calc_pred(x_hat_series, quantised='No'):
    
    '''
    Keyword Arguments:
    ------------------
    x_hat_series -- Aposteriori estimates (real and estimated imaginary components of the state for each basis frequency) for num_of_time_steps [Dim: twonumf x num_of_time_steps. dtype = float64]
    
    Returns:
    ------------------
    pred -- Measurement predictions based on adding the real parts of x_hat [Len: twonumf. dtype = float64]
    '''
    
    series = x_hat_series.shape[2]
    jitter = np.zeros(series)
    for k in xrange(series):
        jitter[k] = np.sum(x_hat_series[::2, 0, k])
        
    if quantised =='No':
        return jitter
    else:
        return projected_msmt(jitter)


#@nb.jit(nopython=True)
def calc_Gamma(x_hat, oe, numf):
    '''Returns a vector of noise features used to calculate Q in Kalman Filtering
       Could be simplified via slicing. Not yet implemented.
    '''
    Gamma2 = np.zeros((2*numf,1))
    spectralresult0=0
    spectralresult=0
    for spectralresult0 in xrange(numf):
        spectralresult = spectralresult0*2
        Gamma2[spectralresult,0] = x_hat[spectralresult,0]*(np.sqrt(oe**2/ (x_hat[spectralresult,0]**2 + x_hat[spectralresult + 1,0]**2)))
        Gamma2[spectralresult+1,0] = x_hat[spectralresult + 1,0]*(np.sqrt(oe**2/ (x_hat[spectralresult,0]**2 + x_hat[spectralresult + 1,0]**2)))   
    return Gamma2


def get_dynamic_model(twonumf, Delta_T_Sampling, freq_basis_array, coswave=-1):
    
    """ Returns the dynamic state space model based 
    on computational basis and experimental sampling params
    [Dim: twonumf x twonumf. dtype = float64]
    """
    
    a = np.zeros((twonumf,twonumf))
    index = range(0,twonumf,2)
    index2 = range(1,twonumf+1,2) # twnumf is even so need to add 1 to write over the last element
    diagonals = np.cos(Delta_T_Sampling*freq_basis_array*2*np.pi) # dim(diagonals) = numf
    off_diagonals = coswave*np.sin(Delta_T_Sampling*freq_basis_array*2*np.pi) # dim(off-diagonals) = numf
    a[index, index] = diagonals
    a[index2, index2] = diagonals
    a[index, index2] = off_diagonals
    a[index2, index] = -1.0*off_diagonals
     
    return a


def propagate_states(a, x_hat, P_hat, oe, numf):
    '''Returns state propagation without a Kalman update. 
    '''
    x_hat_apriori = np.dot(a, x_hat) 
    Gamma = np.dot(a,calc_Gamma(x_hat, oe, numf))
    Q = np.outer(Gamma, Gamma.T)
    P_hat_apriori = np.dot(np.dot(a,P_hat),a.T) + Q

    return x_hat_apriori, P_hat_apriori, Q


def calc_z_proj(h, x_hat_apriori):
    z_proj = np.dot(h, x_hat_apriori)
    return z_proj


def calc_Kalman_Gain(h_, P_hat_apriori, rk, quantised='No', x_hat_apriori=None):
    '''Returns the Kalman gain and scalar S for performing state updates.
    NB: S can be a matrix <=> rk is a matrix, and S_inv = np.linalg.inverse(S) 
    instead of 1.0/S.
    '''
    
    if quantised =='No':
        h = h_ # Linear Measurement  
    else:
        h = -1*np.sin(2.0*calc_z_proj(h_, x_hat_apriori))*h_ # Nonlinear Jacobian eval at current time step
        
    #S = la.multi_dot([h,P_hat_apriori,h.T]) + rk 
    #intermediary = np.dot(P_hat_apriori, h.T)
    #S = np.dot(h, intermediary) + rk 
    #S = np.dot(np.dot(h, P_hat_apriori), h.T) + rk #Agreement between detailed KFs
    
    S = np.dot(h, np.dot(P_hat_apriori, h.T)) + rk #Agreement between fast KFs, same as linalg.multi_dot for associativity problem
    S_inv = 1.0/S # 1.0/S and np.linalg.inv(S) are equivalent when S is rank 1
    
    if not np.isfinite(S_inv).all():
        print("S is not finite")
        raise RuntimeError
    
    W = np.dot(P_hat_apriori, h.T)*S_inv
    return W, S

def one_shot_msmt(n=1, p=0.5, num_samples=1):
    return np.random.binomial(n,p,size=num_samples)
    
def projected_msmt(jitter):
    prob_of_msmt = 0.5 + 0.5*np.cos(2.0*jitter)
    quantised_msmt =[]
    for item in prob_of_msmt:
        quantised_msmt.append(one_shot_msmt(p=item))
    return np.array(quantised_msmt).ravel()

def calc_residuals(h, x_hat_apriori, msmt, quantised='No'):
    jitter = calc_z_proj(h, x_hat_apriori)
    if jitter > np.pi:
        print('Total jitter is: ', jitter)
    if quantised == 'No':
        return msmt - jitter # Linear measurement
    else:
        return msmt - projected_msmt(jitter) # non linear msmt
    

       # if projected_msmt < 0:
       #     print("Saturation:", projected_msmt)
       #     projected_msmt=0.0
       # elif projected_msmt > 1:
       #     print("Saturation:", projected_msmt)
       #     projected_msmt=1.0
       # quantised_msmt = one_shot_msmt(p=projected_msmt)
    
