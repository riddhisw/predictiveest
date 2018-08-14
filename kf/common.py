'''
Created on Thu Apr 20 19:20:43 2017
@author: riddhisw

.. module:: kf.common

    :synopsis: List of common functions applicable to kf.detailed, kf.fast and
        kf.fast_2 module calculations, with a focus on LKFFB.

    Module Level Functions:
    ----------------------
        calc_inst_params :  Return instantaneous amplitudes and instaneous phases
            associated with each LKFFB basis oscillator.
        calc_pred : Return one step ahead predicted observation based on the
            apriori Kalman state.
        calc_Gamma : Return a vector of noise features in LKFFB.
        get_dynamic_model : Return the dynamic state space model for LKFFB.
        propagate_states : Return state propagation using the Kalman dynamic model
            but without a Kalman gain / Bayesian update.
        calc_z_proj : Return the measured output of a Kalman state, under a linearisable
            measurement model.
        calc_Kalman_Gain : Return the Kalman gain and scalar S for performing state
            updates.
        one_shot_msmt : Return a single shot qubit measurement, with Born probability
            for measuring an up state specified as p.
        projected_msmt : Return a qubit measurement outcome based on an estimate of relative
            stochastic qubit phase under dephasing.
        calc_residuals : Return residuals between one step ahead predictions and
            measurement data.

.. moduleauthor:: Riddhi Gupta <riddhi.sw@gmail.com>

'''

from __future__ import division, print_function, absolute_import

import numpy as np
#import numba as nb
import numpy.linalg as la


#@nb.jit(nopython=True)
def calc_inst_params(x_hat_time_slice):
    '''
    Return instantaneous amplitudes and instaneous phases associated with each
    LKFFB basis oscillator using state estimate, x_hat_time_slice, at a given
    time step.

    Parameters:
    ----------
        x_hat_time_slice (`float64`) : Kalman state estimate at a specific time step.

    Returns:
    -------
        instantA_slice (`float64`) : Instantaneous amplitude information from LKFFB.
        instantP_slice (`float64`) : Instantaneous phase information from LKFFB.
    '''

    # Using apostereroiri estimates
    instantA_slice = np.sqrt(x_hat_time_slice[::2, 0]**2 + x_hat_time_slice[1::2, 0]**2)
    # Correct phase using atan2
    instantP_slice = np.arctan2(x_hat_time_slice[1::2, 0], x_hat_time_slice[::2, 0])
    # Changed math.atan2 to numpy.atan2 to support vectoristion.

    return instantA_slice, instantP_slice


#@nb.jit(nopython=True)
def calc_pred(x_hat_series, quantised='No'):

    ''' Return one step ahead predicted observation based on the apriori Kalman state.

    Parameters:
    ----------
        x_hat_series : Aposteriori estimates (real and estimated imaginary components
            of the state for each basis frequency) for num_of_time_steps,
            [dims: twonumf x num_of_time_steps].

    Returns:
    -------
        If quantised == 'No':
            jitter : one step ahead relative qubit phase predictions based on
                adding only the real parts of Kalman sub-states in x_hat_series.

        If quantised == 'Yes':
            msmt : Return 0 or 1 predicted observation, based on jitter information
               in the Kalman sub-states.
    '''
    series = x_hat_series.shape[2]
    jitter = np.zeros(series)
    for k in xrange(series):
        jitter[k] = np.sum(x_hat_series[::2, 0, k])

    if quantised == 'No':
        return jitter
    else:
        return projected_msmt(jitter)


#@nb.jit(nopython=True)
def calc_Gamma(x_hat, oe, numf):
    ''' Return a vector of noise features in LKFFB.

        Noise features (Gamma) vector is a supporting variable to
            the calculation of the process noise covariance matrix in LKFFB.

       Parameters:
       ----------
            x_hat (`flat64`) : Kalman state vector (LKFFB)
            oe (`float64`) : Kalman process noise variance scale.
            numf (`int`) : Number of sub-states in the LKFFB.

       Returns:
       -------
            Gamma2 (`float64`) : LKFFB process noise features vector.

       To Do:
       -----
            Speed up function calls via vectorised slicing.
    '''
    Gamma2 = np.zeros((2*numf, 1))
    spectralresult0 = 0
    spectralresult = 0
    for spectralresult0 in xrange(numf):
        spectralresult = spectralresult0*2
        Gamma2[spectralresult,0] = x_hat[spectralresult,0]*(np.sqrt(oe**2/ (x_hat[spectralresult,0]**2 + x_hat[spectralresult + 1,0]**2)))
        Gamma2[spectralresult+1,0] = x_hat[spectralresult + 1,0]*(np.sqrt(oe**2/ (x_hat[spectralresult,0]**2 + x_hat[spectralresult + 1,0]**2)))   
    return Gamma2


def get_dynamic_model(twonumf, Delta_T_Sampling, freq_basis_array, coswave=-1):
    '''
    Return the dynamic state space model for LKFFB, based on computational basis and
    experimental sampling params.

    Parameters:
    ----------
        twonumf (`int`):  Twice the number of sub-states in LKFFB, to track
            real and imaginary part of each sub-state independently.
        Delta_T_Sampling (`float64`):  Time between measurements and
            analysis_tools.experiment.Delta_T_Sampling attribute.
        freq_basis_array (`float64`):  Frequency basis for LKFFB.
            See also: analysis_tools.kalman.basis_list attribute for built-in
            LKFFB basis options.
        coswave (`float64`):  Design choice for dynamical model, given by `a`, for LKFFB.
            Defaults to -1.


    Returns:
    -------
        a (`float64`):  Dynamical state space model for LKFFB [dims: twonumf x twonumf].
    '''
    a = np.zeros((twonumf, twonumf))
    index = range(0, twonumf, 2)

    # twnumf is even so need to add 1 to write over the last element
    index2 = range(1, twonumf+1, 2)

    # dim(diagonals) == numf
    diagonals = np.cos(Delta_T_Sampling*freq_basis_array*2*np.pi)

    # dim(off-diagonals) == numf
    off_diagonals = coswave*np.sin(Delta_T_Sampling*freq_basis_array*2*np.pi)
    a[index, index] = diagonals
    a[index2, index2] = diagonals
    a[index, index2] = off_diagonals
    a[index2, index] = -1.0*off_diagonals

    return a


def propagate_states(a, x_hat, P_hat, oe, numf):
    '''Return state propagation using the Kalman dynamic model but without a
          a Kalman gain / Bayesian update.

    Parameters:
    ----------
        a (`float64`): Kalman dynamical model for evolution of x_hat.
        x_hat (`float64`): Kalman state vector (posterior at previous time step).
        P_hat (`float64`): Kalman state covariance matrix (posterior at previous time step).
        oe (`float64`): Kalman process noise variance scale.
        numf (`int`):  Number of Kalman sub-states in LKFFB.

    Returns:
    -------
        x_hat_apriori : Kalman state vector (prior at current time step).
        P_hat_apriori : Kalman state covariance matrix (prior at current time step).
        Q : Kalman process noise covariance matrix (due to white noise input
            at current time step).
    '''
    x_hat_apriori = np.dot(a, x_hat)
    Gamma = np.dot(a, calc_Gamma(x_hat, oe, numf))
    Q = np.outer(Gamma, Gamma.T)
    P_hat_apriori = np.dot(np.dot(a, P_hat), a.T) + Q

    return x_hat_apriori, P_hat_apriori, Q


def calc_z_proj(h, x_hat_apriori):
    ''' Return the measured output of a Kalman state, under a linearisable
    measurement model.

    Parameters:
    ----------
        h (`float64`) : Kalman measurement model (linearised representation).
            If measurement model is non-linear, `h` is interpreted as the
            Jacobian of the measurement model and the linearisation is valid as
            a Taylor expansion i.e. valid for small` time-steps.

        x_hat_apriori (`float64`):  Kalman state vector.

    Returns:
    -------
        z_proj (`type`):  Measurement output from measuring the Kalman
            state vector using a linearised measurement model.
    '''
    z_proj = np.dot(h, x_hat_apriori)
    return z_proj


def calc_Kalman_Gain(h_, P_hat_apriori, rk, quantised='No', x_hat_apriori=None):
    '''Return the Kalman gain and scalar S for performing state updates.

    Parameters:
    ----------
        h_ (`float64`) : Kalman measurement model.
        P_hat_apriori (`float64`) : Kalman state variance matrix.
        rk (`float64`) : Kalman measurement noise variance scale.
        quantised (`str`) : 'Yes' / No' flag on measurement model regime:
            If quantised is 'Yes':
                Filtering occurs on single shot, binary qubit outcomes. A non
                linear measurement model for h_ applies.
            If quantised is 'No':
                Filtering occurs on pre-processed qubit measurements. A
                linear measurement model for h_ applies.
            Defaults to 'No'.
            'Yes' functionality -- [DEPRECIATED]
                See Instead: qif package.
        x_hat_apriori (`float64`) : Kalman state vector to be measured.
            Defaults to None. If quantised is 'Yes', then x_hat_apriori cannot
            be None as  x_hat_apriori is required to calculate state-dependent
            Jacobian, h_.

    Returns:
    -------
        W : Kalman gain / Bayesian update for Kalman state estimates.
        S : Intermediary covariance matrix for calculating Kalman gain.
            NB: S can be a matrix iff rk is a matrix, and S_inv = np.linalg.inverse(S)
            instead of 1.0/S.
    '''

    if quantised == 'No':
        h = h_ # Linear Measurement
    else:
        # Nonlinear Jacobian eval at current time step
        h = -1*np.sin(2.0*calc_z_proj(h_, x_hat_apriori))*h_


    # S = la.multi_dot([h,P_hat_apriori,h.T]) + rk
    # intermediary = np.dot(P_hat_apriori, h.T)
    # S = np.dot(h, intermediary) + rk

    # Agreement between detailed KFs
    # S = np.dot(np.dot(h, P_hat_apriori), h.T) + rk 

    # Agreement between fast KFs, same as linalg.multi_dot for associativity problem

    S = np.dot(h, np.dot(P_hat_apriori, h.T)) + rk

    S_inv = 1.0/S # 1.0/S and np.linalg.inv(S) are equivalent when S is rank 1

    if not np.isfinite(S_inv).all():
        print("S is not finite")
        raise RuntimeError

    W = np.dot(P_hat_apriori, h.T)*S_inv
    return W, S

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
    return np.random.binomial(n,p,size=num_samples)


def projected_msmt(jitter):
    '''Return a qubit measurement outcome based on an estimate of relative stochastic
    qubit phase under dephasing.'''

    prob_of_msmt = 0.5 + 0.5*np.cos(2.0*jitter)
    quantised_msmt = []

    for item in prob_of_msmt:
        quantised_msmt.append(one_shot_msmt(p=item))

    return np.array(quantised_msmt).ravel()

def calc_residuals(h, x_hat_apriori, msmt, quantised='No'):
    '''Return residuals between one step ahead predictions and measurement data.

    Parameters:
    ----------
        h ('float64') : Kalman measurement model.
        x_hat_apriori ('float64') : Kalman state vector.
        msmt ('float64') : Input measurement data:
            If quantised is 'Yes':
                Filtering occurs on single shot, binary qubit outcomes. Measurements
                are quantised (binary outcomes) representing qubit state as up or down.
            If quantised is 'No':
                Filtering occurs on pre-processed qubit measurements. Measurements are
                floating point numbers representing relative qubit phase under
                dephasing.
        quantised (`str`, optional) : 'Yes' / No' flag on measurement model regime:
            If quantised is 'Yes':
                Filtering occurs on single shot, binary qubit outcomes. Residuals
                are quantised (binary outcomes) representing error in actual qubit measurement
                and the Kalman one-step ahead predicted qubit measurement.
            If quantised is 'No':
                Filtering occurs on pre-processed qubit measurements. Residuals are
                floating point numbers representing error in qubit phase (via pre-processed qubit
                measurement outcomes) and the Kalman one-step ahead qubit phase prediction.
            Defaults to 'No'.
            'Yes' functionality -- [DEPRECIATED]
                See Instead: qif package.
    '''
    jitter = calc_z_proj(h, x_hat_apriori)

    # if jitter > np.pi:
    #    print('Total jitter is: ', jitter)

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
