'''
Created on Thu Apr 20 19:20:43 2017
@author: riddhisw

.. module:: qif.common

    :synopsis: Lists functions required for the analysis and execution of a Quantised
        Kalman Filter (QKF).

    Module Level Functions:
    ----------------------
        noisy_z : Return noisy simulated measurment outcomes by making measurements
            on evolution of an internal (hidden) state.
        projected_msmt : Return a qubit measurement outcome based on an estimate
            of relative stochastic qubit phase under dephasing.
        one_shot_msmt : Return a single shot qubit measurement, with Born probaility
            for measuring an up state specified as p.
        generate_AR : Return a num-length autoregressive (AR) sequence of order q.
        calc_h : Return Kalman state-dependent measurement model h(x).
        calc_H : Return Jacobian matrix d/dx h(x) where h(x) is a non linear measurement
            model and order refers to AR(order) process.
        propagate_x : Return x_hat_apriori i.e. state propagation without a Kalman update.
        propagate_p : Return P_hat_apriori i.e. state covariance propagation
            without a Kalman update.
        calc_gain : Return the Kalman gain and scalar S for performing state updates,
            with linearised, state-dependent h(x) as measurement model.
        saturate : Saturates p between [-threshold, threshold] for a one bit
            quantiser [Helper function].
        qkf_state_err :  Return state estimates from QKF output [Helper function].
        normalise : Return normalised input state vector. [Helper function].

.. moduleauthor:: Riddhi Gupta <riddhi.sw@gmail.com>
'''

from __future__ import division, print_function, absolute_import

import numpy as np
import numpy.linalg as la
import scipy.stats as stats
from scipy.special import erf as erf_func


############################################### QIF Bayes Risk Helper Funcs ####

def qkf_state_err(x_states, truths):
    '''Return state estimates from QKF output [Helper function].'''

    errs = (x_states - truths)**2
    avg_err_sqr = np.mean(errs, axis=0)

    return avg_err_sqr

def normalise(x):
    '''Return normalised input state vector. [Helper function].'''
    norm = np.linalg.norm(x)
    if norm != 0.:
        return x / norm

############################################### AR PROCESS DATA ################


def generate_AR(xinit, num, weights, oe):
    ''' Return a num-length AR sequence of order q.

    Parameters:
    ----------
        xinit : Initial state vector, where xinit.shape[0] == weights.shape[0].
        num : Total length of the output vector (num of forward steps = num - weights.shape[0]).
        weights : AR coefficients, where order of AR process == weights.shape[0].
        oe : White noise variance (Kalman equivalent of process noise variance scale.)

    Returns:
    -------
        x : An AR process of order == weights.shape[0] with total number of
            terms == num.
    '''

    x = np.zeros(num)
    order = weights.shape[0]

    x[0: order] = xinit

    for step in range(order, num):

        for idx_weight in xrange(order):

            x[step] += weights[idx_weight]*x[step - 1 - idx_weight]

        x[step] +=  np.random.normal(scale=np.sqrt(oe))

    return x

############################################### QUANTISATION MODEL #############

def one_shot_msmt(n=1, p=0.5, num_samples=1):

    '''Return a single shot qubit measurement, with Born probaility for measuring an up
        state specified as p.

    Parameters:
    ----------
        n (`int`):  Total trials to define a binomial distribution. For a single qubit
            measurement, we set n == 1 as a Bernoulli trial.
        p (`float64`):  Sucess probability for measuring the qubit in an up state.
            Default set to 0.5.
        num_samples (`int`): Number of samples drawn from binomial distribution.
            Default set to 1.

    Returns:
    -------
        Qubit state (`int`):  Return 0 or 1 outcome from a Bernoulli trial.
    '''
    return np.random.binomial(n, p, size=num_samples)


def saturate(p_, threshold=0.5):
    ''' Saturates p between [-threshold, threshold] for a one bit quantiser.
    [Helper function].
    '''

    p = np.asarray(p_).ravel()

    for n in xrange(p.shape[0]):
        if p[n] > threshold:
            p[n] = threshold
        if p[n] < -1.* threshold:
            p[n] = -1.* threshold
    return p


def projected_msmt(z_proj):
    ''' Return a qubit measurement outcome based on an estimate of relative stochastic
    qubit phase under dephasing.

    Parameters:
    ----------
        z_proj (`float`) : Outcome of measuring a Kalman state vector,
                            according to a Kalman measurement model.

    Returns:
    -------
        Return 0 or 1 measurement based on internal variable, z_proj.
    '''

    quantised_msmt = []

    for item in z_proj:

        # # Linear Model without Quantisation
        # if item <0 :
        #     quantised_msmt.append(-1.0)
        # elif item >= 0:
        #     quantised_msmt.append(1.0)

        # # Linear Msmt Model with Amp Quantisation
        # bias = 0.5*saturate(item, threshold=1.0) + 0.5
        # quantised_msmt.append(one_shot_msmt(p=bias)*2.0 - 1.0)

        # Non Linear Msmt Model with Amp Quantisation
        bias = saturate(item, threshold=0.5) + 0.5
        quantised_msmt.append(one_shot_msmt(p=bias))

    # # Turn off quantisation
    # return z_proj
    return np.asarray(quantised_msmt).ravel()


############################################### MEASUREMENT MODEL ##############

def noisy_z(x, rk, saturate_='Yes'):

    ''' Return noisy simulated measurment outcomes by making measurements
        on evolution of an internal (hidden) state, x, via a (non-linear)
        measurement model and subject to Gaussian white measurement noise.

    Parameters:
    ----------
        x (`float64`):  Input random process that cannot be observed without
            incurring additional white Gaussian zero mean measurement noise.
        rk (`float64`):  Measurement noise variance scale.
        saturate_ (`str`) : 'Yes'/ 'No' flag to saturate values of output random
            variable within certain bounds. 'Yes' will add saturation errors.

    Returns:
    -------
        z (`float64`):  Output random process obtained by measuring
            x according to a known measurement model, and further
            corrupted by measurement noise.
    '''
    z = np.zeros(x.shape[0])

    # # Linear Msmt Model
    # z[:] = x[:]

    # Non Linear Msmt Model
    z = 0.5*np.cos(x)

    if rk != 0.0:
        print(rk)
        z += np.random.normal(loc=0.0, scale=np.sqrt(rk), size=z.shape[0])

    if saturate_ == 'No':
        return z

    # # Linear Msmt Model without Quantisation
    # saturated_z = 1.0*z

    # # Linear Msmt Model with Amp Quantisation
    # saturated_z = saturate(z, threshold=1.0)

    # Non Linear Msmt Model
    saturated_z = saturate(z, threshold=0.5)
    return saturated_z

def calc_h(x_hat_apriori):
    ''' Return Kalman state-dependent measurement model h(x)'''

    # # Linear Msmt Model
    # h = x_hat_apriori[0]

    # # Non Linear Msmt Model
    h = 0.5*np.cos(x_hat_apriori[0])
    return h

def calc_H(x_hat_apriori_):
    ''' Return Jacobian matrix d/dx h(x) where h(x) is a non linear measurement model and order
    refers to AR(order) process.

    h(x[0]) = 0.5 + 0.5 * cos(x[0]) and 0 elsewhere x[1:]
    H \equiv  -0.5 * sin(x[0]) '''

    x_hat_apriori = np.asarray(x_hat_apriori_).ravel()
    order = x_hat_apriori.shape[0]
    H = np.zeros(order)

    # # Linear Msmt Model
    # H[0] = 1.0

    # # Non Linear Msmt Model
    H[0] = -0.5*np.sin(x_hat_apriori[0])

    return H

############################################### KALMAN FILTERING ###############

def propagate_x(a, x_hat):
    '''Return x_hat_apriori i.e. state propagation without a Kalman update.

    Parameters:
    ----------
        a (`float64`):  Kalman dynamical model for propagating states between time steps.
        x_hat (`float64`):  Kalman state vector (posterior at previous time step).


    Returns:
    -------
        x_hat_apriori (`float64`):  Kalman state vector (prior at current time step).
    '''
    return np.dot(a, x_hat)

def propagate_p(a, P_hat, Q):
    '''Return P_hat_apriori i.e. state covariance propagation without a Kalman update.

    Parameters:
    ----------
        a (`float64`):  Kalman dynamical model for propagating states between time steps.
        P_hat (`float64`):  Kalman state covariance matrix (posterior at previous time step).
        Q (`float64`): Process noise covariance matrix (noise injection at current time step).


    Returns:
    -------
        P_hat_apriori (`float64`):  Kalman covariance matrix (prior at current time step).
    '''
    return np.dot(np.dot(a, P_hat),a.T) + Q

def calc_gain(x_hat_apriori, P_hat_apriori, rk):
    ''' Return the Kalman gain and scalar S for performing state updates,
        with linearised, state-dependent h(x) as measurement model.

    Parameters:
    ----------
        P_hat_apriori (`float64`) : Kalman state variance matrix.
        rk (`float64`) : Kalman measurement noise variance scale.
        x_hat_apriori (`float64`) : Kalman state vector required for
            defining a state-dependent measurement model.

    Returns:
    -------
        W : Kalman gain / Bayesian update for Kalman state estimates.
        S : Intermediary covariance matrix for calculating Kalman gain.
            NB: S can be a matrix iff rk is a matrix, and S_inv = np.linalg.inverse(S)
            instead of 1.0/S.
    '''

    # Jacobian of msmt model
    h = calc_H(x_hat_apriori)

    # # Additional variance from quantisation, see Karlsson (2005) for \Delta = 2, m=1.
    S = np.dot(h, np.dot(P_hat_apriori, h.T)) + rk + (2**2 / 12)

    S_inv = 1.0/S

    if not np.isfinite(S_inv).all():
        print("S is not finite")
        raise RuntimeError

    W = np.dot(P_hat_apriori, h.T)*S_inv
    return W, S

def update_p(P_hat_apriori, S, W):
    ''' Return state covariance update calculation. [Helper Function].'''
    return  P_hat_apriori - S*np.outer(W, W.T)

def calc_residuals(prediction, msmt):
    '''Return residuals between incoming msmt data and quantised predictions [Helper Function].'''
    return msmt - prediction

def calc_z_proj(x_hat_apriori):
    '''Return projected one step ahead measurement with h(x) msmt model. [Helper Function].'''
    return calc_h(x_hat_apriori)