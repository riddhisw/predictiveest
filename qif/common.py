from __future__ import division, print_function, absolute_import

import numpy as np
import numpy.linalg as la
import scipy.stats as stats
from scipy.special import erf as erf_func

############################################### QIF Bayes Risk Helper Funcs ########################################

def qkf_state_err(x_states, truths):
    '''Returns state estimates from QKF output'''
    
    errs = (x_states - truths)**2
    avg_err_sqr = np.mean(errs, axis=0)
    
    return avg_err_sqr

def normalise(x):
    norm = np.linalg.norm(x)
    if norm != 0.:
        return x / norm

############################################### AR PROCESS DATA ########################################


def generate_AR(xinit, num, weights, oe):
    ''' Returns a num-length AR sequence of order q = weights.shape[0] and process noise variance, oe, and initial conditions vector xinit, where xinit.shape[0] == weights.shape[0]'''
    
    x = np.zeros(num)
    order = weights.shape[0]
    
    x[0: order] = xinit
    
    for step in range(order, num):

        for idx_weight in xrange(order):
        
            x[step] += weights[idx_weight]*x[step - 1 - idx_weight]

        x[step] +=  np.random.normal(scale=np.sqrt(oe)) 
        
    return x
    

############################################### QUANTISATION MODEL ########################################


def one_shot_msmt(n=1, p=0.5, num_samples=1):
    '''Returns a one-bit quantised [0,1] i.e. a biased coin is flipped with bias, p.'''
    return np.random.binomial(n,p,size=num_samples) 


def saturate(p_, threshold=0.5):
    ''' Saturates p between [-threshold, threshold] for a one bit quantiser'''
    
    p = np.asarray(p_).ravel()
    
    for n in xrange(p.shape[0]):
        if p[n] > threshold:
            p[n] = threshold
        if p[n] < -1.* threshold:
            p[n] = -1.* threshold
    
    return p

  
def projected_msmt(z_proj):
    ''' Returns a quantised [0 or 1] outcome for biases given in z_proj'''
    quantised_msmt =[]
    
    for item in z_proj:
    
        # Alternative Linear Model without coin flip quantisation - this only effects RMSE of QIF, not CRLB. If coin flipping is used, additional randomness effects QIF trajectories. If no coin flipping is used, then QIF trajectories are identical. Sign quantised CRLB recursion is unaffected in the linear H, scalar Kalman state regime.
        
        #if item <0 :
        #    quantised_msmt.append(-1.0)
        #elif item >= 0:
        #    quantised_msmt.append(1.0)
        
        # bias = 0.5*saturate(item, threshold=1.0) + 0.5 # Linear Msmt Model with Amp Quantisation
        # quantised_msmt.append(one_shot_msmt(p=bias)*2.0 - 1.0) # Linear Msmt Model with Amp Quantisation
        
        bias = saturate(item, threshold=0.5) + 0.5 # Non Linear Msmt Model with Amp Quantisation
        quantised_msmt.append(one_shot_msmt(p=bias)) # Non Linear Msmt Model with Amp Quantisation
        
    return np.asarray(quantised_msmt).ravel() # Turn off quantisation by return z_proj


############################################### MEASUREMENT MODEL ########################################

def noisy_z(x, rk, saturate_='Yes'):

    '''Generates noisy z from a sequence of true x'''
    z = np.zeros(x.shape[0])
    # z[:] = x[:] # Linear Msmt Model
    z = 0.5*np.cos(x) # Non Linear Msmt Model 
    
    if rk !=0.0:
        print(rk)
        z += np.random.normal(loc=0.0, scale=np.sqrt(rk), size=z.shape[0]) # not equivalent to rk*np.random.normal(size=z.shape[0]) 
    
    if saturate_ == 'No':
        return z
    
    # saturated_z = 1.0*z # Alternative Linear Msmt Model without coin flip quantisation
    # saturated_z = saturate(z, threshold=1.0) # Linear Msmt Model
    saturated_z = saturate(z, threshold=0.5) # Non Linear Msmt Model
    
    return saturated_z

def calc_h(x_hat_apriori):
    ''' Returns h(x)'''
    # h = x_hat_apriori[0] # Linear Msmt Model
    h = 0.5*np.cos(x_hat_apriori[0]) # Non Linear Msmt Model
    return h

def calc_H(x_hat_apriori_):
    ''' Returns Jacobian matrix d/dx h(x) where h(x) is a non linear measurement model and order
    refers to AR(order) process. 
    
    h(x[0]) = 0.5 + 0.5 * cos(x[0]) and 0 elsewhere x[1:]
    H \equiv  -0.5 * sin(x[0]) '''
    
    x_hat_apriori = np.asarray(x_hat_apriori_).ravel()
    order = x_hat_apriori.shape[0]
    H = np.zeros(order)
    
    # H[0] = 1.0 # Linear Msmt Model
    H[0] = -0.5*np.sin(x_hat_apriori[0]) # Non Linear Msmt Model
    
    # print('H_n shape', H.shape, H) # 'j_onebit', j_onebit)
    return H

############################################### KALMAN FILTERING #########################################

def propagate_x(a, x_hat):
    '''Returns x_hat_apriori i.e. state propagation without a Kalman update. 
    '''
    return np.dot(a, x_hat) 

def propagate_p(a, P_hat, Q):
    '''Returns P_hat_apriori i.e. state covariance propagation without a Kalman update. 
    '''
    return np.dot(np.dot(a, P_hat),a.T) + Q  
   
def calc_gain(x_hat_apriori, P_hat_apriori, rk):
    '''Returns the kalman gain with linearised h(x) as msmt model
    '''
    
    h = calc_H(x_hat_apriori) # jacobian of msmt model
    S = np.dot(h, np.dot(P_hat_apriori, h.T)) + rk + (2**2 / 12) # additional variance from quantisation, see Karlsson (2005) for \Delta = 2, m=1.
    S_inv = 1.0/S
    
    if not np.isfinite(S_inv).all():
        print("S is not finite")
        raise RuntimeError
    
    W = np.dot(P_hat_apriori, h.T)*S_inv
    return W, S

    
def update_p(P_hat_apriori, S, W):
    return  P_hat_apriori - S*np.outer(W,W.T)

def calc_residuals(prediction, msmt):
    '''Returns residuals between incoming msmt data and quantised predictions'''
    return msmt - prediction
    
    
def calc_z_proj(x_hat_apriori):
    '''Returns projected one step ahead measurement with h(x) msmt model
    '''
    return calc_h(x_hat_apriori) 
    
 
############################################### QUANTISED ONE BIT CRLB WITH TIME-VARYING H (V, S, Q, R are time invariant) ########################################
# Refer Karlsson (2005) for notation

def rho(z_value, rk, b =0.5):
    import math
    
    ''' Returns probability of quantised outcome, y, given z,  i.e. p(y|z), where y = z + e.
    Measurement error e is zero mean, white and Gaussian distributed with variance rk (not standard deviation) . 
    In Karrlson (2005), p(y|z) \equiv Pr.(e < -z) \equiv Pr (e' < -z/ rk) for zero mean Gaussian normal e'''
    
    error_dist = stats.norm(loc=0, scale=1)
    normalised_value = -1*(z_value - 0.0) /np.sqrt(rk) # normalised via standard deviation, take out 0.5 pi mean in x makes z zero mean
    rho = error_dist.cdf(normalised_value) # Theorum 3 
    
    # rho = (1./(4*np.sqrt(rk*np.pi)))*(1 + z_value)*(math.erf(z_value + b) + math.erf(b -z_value))
    # rho += (1./(4*np.pi)) * ( np.exp( -(1./rk)*(z_value + b)**2 ) - np.exp( -(1./rk)*(z_value - b)**2 )  )    
    
    return rho

def J_one_bit(z_value, rk):
    
    '''Returns J_{m=1, t} given by (64) for the calculation in (41) in Karlsson (2005)'''
    j_onebit = np.exp(-(z_value**2)/rk)*( 1./((rho(z_value, rk) * (1. - rho(z_value, rk))))) / (2.0 * np.pi*rk) # Theorum 3
    # j_onebit = 1.0 / (rho(z_value, rk)*(1.0 - rho(z_value, rk)))
    return j_onebit
    
    
def J_x_n(x_value, z_value, rk):
    
    '''Returns J(x_n)_n for a single time step in (41) to enable recursion of covariance in (50) in Karlsson (2005), 
    using time varying measurement matrix H_n(x_n), true observed state z_n, and measurement noise variance rk. 
    We are concerned with only the first element of x_hat_posteriori[0] = x_value i.e H[1:] \equiv 0. 
    Hence, we treat this as a scalar recursion'''
    j_onebit = J_one_bit(z_value, rk)    
    H_n = calc_H(x_value) # measurement model chosen in calc_H() function manually! this function outputs a Jacobian matrix.
    
    # print('H_n shape', H_n.shape) # 'j_onebit', j_onebit)
    
    j_x_n =  j_onebit*np.outer(H_n, H_n) # np.dot(np.dot(np.transpose(H_n), j_onebit), H_n) # for J(x, m=1) scalar, using outer() specifies array shape.
    
    return j_x_n

def inverse_Ricatti_recursion(x_hat_, z_hat_, rk, true_oe, dynamical_model, time_steps=100, p0=0.0):
    '''Returns Ricatti covariance via recursion for a one bit quantiser. 
    Note that Q_inv, V, S, are time invariant for our model. 
    Only J(x) changes. These terms are defined in Karlsson (2005)'''
    
    Q_inv = np.eye(dynamical_model.shape[0])* (1.0 / true_oe**2) # Not Q in AKF, but approx defined to have an inverse
    V = np.dot(np.dot(dynamical_model, Q_inv), dynamical_model.T)
    S = -1 * np.dot(dynamical_model.T, Q_inv)
    
    # print('Q_inv', Q_inv, 'V', V, 'S', S)
    # print('dynamical model', dynamical_model)
    
    order = dynamical_model.shape[0] 
    # posterior = np.zeros((order, order, time_steps)) # CHANGED
    posterior = np.zeros((time_steps)) # CHANGED
    J_empty = np.zeros((order, order))
    J_empty[0,0] = 1.0
    # posterior[:,:, 0] = p0 # CHANGED
    posterior[0] = p0 # NOTE: THIS IS INVERSE OF VARIANCE> SMALL NUMBERS == LARGE VARIANCE
    
    for n in range(1, time_steps): # effectively implement a scalar recursion for f_n (not x) since Q, V, S, are time invariant
        
        # posterior[: , :, n] = Q_inv + J_x_n(x_hat_[n], z_hat_[n], rk).ravel()*J_empty # Fisher info calculated at the same time step as posterior # CHANGED
        # update = np.linalg.inv(posterior[: , :, n-1] + V)  # CHANGED
        # posterior[: , :, n] += -1 * np.dot(np.dot(S.T, update), S) # CHANGED
        
        posterior[n] = Q_inv[0,0] + J_x_n(x_hat_[n], z_hat_[n], rk)[0,0]
        update = 1./(posterior[ n-1] + V[0,0]) 
        posterior[n] += -1 * update * S[0,0]**2
        
        if not np.isfinite(posterior).all():
            print("posterior is not finite")
            print('J', J_x_n(x_hat_[n], z_hat_[n], rk))
            print('Q_inv', Q_inv)
            print('S', S)
            print('dynamical_model', dynamical_model)
            raise RuntimeError
    
    # print('J', J_x_n(x_hat_[n], z_hat_[n], rk).ravel()*J_empty)
    # print('shape of J', J_x_n(x_hat_[n], z_hat_[n], rk)[0,0])
    # print('Shape of Q', Q_inv[0,0])
    # print('Shape of S', S[0,0])
    
    return posterior

############################################### QUANTISED COINFLIP ONE BIT CRLB WITH TIME-VARYING H (V, S, Q, R are time invariant) ########################################


def J_one_bit_coinflip(z_value, rk):
    
    '''Returns J_{m=1, t} given by derived coin flip msmt action'''
    
    scale_rk = np.real(np.sqrt(2*rk))
    rho_0 = erf_func(1.0/(scale_rk)) + (scale_rk/np.sqrt(np.pi))*(np.exp(-(1.0/scale_rk)**2)- 1.0)
    
    if abs(z_value)== 0.5:
        print("diverged, reset z = 0.49999999") # avoids divergence at the boundaries
        z_value=0.49999999 # INCORRECT - doesnt account for negative values
    
    j_onebit = (rho_0 * 4.0 ) / (1.0 - 4*(z_value)**2)
    return j_onebit
    
    
def J_x_n_coinflip(x_value, z_value, rk):
    
    '''identical to J_x_n but calls on J_one_bit_coinflip'''
    
    j_onebit = J_one_bit_coinflip(z_value, rk)    
    H_n = calc_H(x_value)
    j_x_n =  j_onebit*np.outer(H_n, H_n)
    
    return j_x_n

def inverse_Ricatti_recursion_coinflip(x_hat_, z_hat_, rk, true_oe, dynamical_model, time_steps=100, p0=0.0):
    '''identical to inverse_Ricatti_recursion but calls on J_x_n_coinflip'''
    
    Q_inv = np.eye(dynamical_model.shape[0])* (1.0 / true_oe**2) 
    V = np.dot(np.dot(dynamical_model, Q_inv), dynamical_model.T)
    S = -1 * np.dot(dynamical_model.T, Q_inv)  
    
    order = dynamical_model.shape[0] 

    posterior = np.zeros((time_steps)) 
    J_empty = np.zeros((order, order))
    J_empty[0,0] = 1.0

    posterior[0] = p0 # 
    
    for n in range(1, time_steps): 
        
        posterior[n] = Q_inv[0,0] + J_x_n_coinflip(x_hat_[n], z_hat_[n], rk)[0,0]
        update = 1./(posterior[ n-1] + V[0,0]) 
        posterior[n] += -1 * update * S[0,0]**2
        
        if not np.isfinite(posterior).all():
            print("posterior is not finite")
            print('J', J_x_n(x_hat_[n], z_hat_[n], rk))
            print('Q_inv', Q_inv)
            print('S', S)
            print('dynamical_model', dynamical_model)
            raise RuntimeError
    
    return posterior
############################################### Scalar Time Invariant Discrete Time Riccati Steady State ########################################

def scalar_dre_time_invariant_steady_state(Q, H, R, Phi, p0, k):
    

    s1 = np.sqrt((H**2)*Q + R*((Phi + 1)**2))
    s2 = np.sqrt((H**2)*Q + R*((Phi - 1)**2))
    sigma = s1 * s2
    
    tau_1 = (H**2)*Q + R*(Phi**2 - 1) + sigma
    tau_2 = (H**2)*Q + R*(Phi**2 - 1) - sigma
    
    
    
    lambda2 = ((H**2)*Q + R*(Phi**2 + 1) - sigma) #/ (2*Phi*R)
    lambda1 = ((H**2)*Q + R*(Phi**2 + 1) + sigma) # / (2*Phi*R)
    rho_k = (lambda2 / lambda1)**k
    
    top = p0*tau_2 + 2*Q*R - rho_k*(p0*tau_1 + 2.*Q*R)
    bottom = 2*(H**2)*p0 - tau_1 - rho_k*( 2*(H**2)*p0 - tau_2)
    
    P = top / bottom
    
    return P


def scalar_time_inv_sign_quantiser_steady_state(Q, R, F):

    ''' Returns non complex steady state solns to  P for scalar, time invariant, linear H \equiv 1 sign quantiser for Q = var(process noise), R = var(msmt noise), F = AR(1) coefficient by solving a quadratic equation in P, as defined in Example 7 of Karlsson
    '''

    # calculate coefficients for quadratic in steady state soln P
    
    a = 1.0
    J = 2.0/ (np.pi * R)
    b = (Q*J + 1. - F**2)/(J * (F**2))
    c = -1*Q/(J * (F**2))
    
    # solve quadratic in Example 7 of Karlsson 2005

    d = b**2 - 4.0*a*c
    
    if d < 0:
        print('No real solutions to steady state P - returning zero')
        return 0.0
        
    P1 = (-b + np.sqrt(d)) / (2 * a)
    P2 = (-b - np.sqrt(d)) / (2 * a)

    return P1, P2

