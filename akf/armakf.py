'''
Created on Thu Apr 20 19:20:43 2017
@author: riddhisw

.. module:: akf.armakf

    :synopsis: Implements Kalman Filtering using autoregressive (AR) dynamics.

    akf.armakf implements Kalman Filtering where dynamics are represented by an
        autoregressive (AR) process of order (q).

    Module Level Functions:
    ----------------------
    get_autoreg_model : Return the dynamic state space model for AR(q).
    propagate_states_no_gamma : Return state propagation without a Kalman update.
    autokf : Save .npz file ooutput from an autoregressive Kalman Filtering (AKF) run.

.. moduleauthor:: Riddhi Gupta <riddhi.sw@gmail.com>
'''

from __future__ import division, print_function, absolute_import

import numpy as np
import sys
sys.path.append('../')

from kf.common import calc_residuals, calc_Kalman_Gain, projected_msmt

def get_autoreg_model(order, weights):
    """ Return the dynamic state space model for AR(q) process.

    Parameters:
    ----------
        order (`int`) : order q of an AR(q) process [Dim:  1x1].
        weights :  coefficients of an AR(q) process [Dim: 1 x order].

    Returns:
    -------
        a (`float64`):  state space dynamics for AR(q) process in AKF [Dim: order x order].
    """
    a = np.zeros((order, order))

    # Allocate weights
    a[0, :] = weights
    # Pick off past values
    idx = range(order-1)
    idx2 = range(1, order, 1)
    a[idx2, idx] = 1.0

    return a


def propagate_states_no_gamma(a, x_hat, P_hat, Q):

    '''Return state propagation without a Kalman update and identity process noise
    features.

    Parameters:
    ----------
         a (`float64`) :  state space dynamics for AR(q) process in AKF [Dim: order x order].
         x_hat (`float64`) : Kalman state vector [Dim: 1 x order].
         P_hat (`float64`) : Kalman state variance [Dim: order x order].
         Q (`float64`) : Kalman process noise variance [Dim: order x order].


    Returns:
    -------
        Returns Kalman state, x_hat_apriori, and state variance, P_hat_apriori
        after propagating Kalman state according to dynamical model and process
        noise variance.

    '''
    x_hat_apriori = np.dot(a, x_hat)
    P_hat_apriori = np.dot(np.dot(a, P_hat), a.T) + Q

    return x_hat_apriori, P_hat_apriori


def autokf(descriptor, y_signal, weights, oe, rk, n_train=1000, n_testbefore=50,
           n_predict=50, p0=10000, skip_msmts=1,  save='No', quantised='No'):

    '''
    Save .npz file ooutput from an autoregressive Kalman Filtering (AKF) run. KF
        parameters are defined in LKFFB (e.g. kf.fast) and are not repeated here.

    Parameters:
    ----------
        descriptor (`str`) : Filename for .npz output of AKF.
        y_signal (`float64`) : Time stamped measurement sequence [Dim: 1 x num ].
        weights (`float64`) : Coefficients of an AR(q) process  [Dim: 1 x order].
        oe (`float64`) : Kalman process noise variance parameter [Dim: 1 x 1].
        rk (`float64`) : Kalman measurement noise variance parameter [Dim: 1 x 1].
            Defaults to 1000.
        n_train (`int`, optional) : Number of measurements during state estimation [Dim: 1 x 1].
        n_testbefore (`int`, optional) : Number of time-steps before training ceases [Dim: 1 x 1].
            Defaults to 50.
        n_predict (`int`, optional) : Number of time-steps after training ceases [Dim: 1 x 1].
            Defaults to 50.
        p0 (`float64`, optional) : Initial state variance [Dim: 1 x 1].
            Defaults to 10000.
        skip_msmts (`int`, optional): Number of measurements - 1 to skip.
            Defaults to 1.
        save (`str`, optional): Saves AKF output as .npz file if 'Yes'.
            Defaults to 'No'.
        quantised (`str`, optional): Implements a QKF (non-linear) measurement model if 'Yes'.
            Defaults to 'No'. [DEPRECIATED - see module QIF instead.]

    Returns:
    -------
        Saves results of Kalman Filtering with AR dynamics as .npz file.

    See Also:
    --------
        qif.qif : Kalman Filter with AR dynamics and non-linear
            measurement model for acting on binary measurements.
    '''

    num = y_signal.shape[0]
    order = weights.shape[0]

    e_z = np.zeros(num)

    idx = range(order)
    h = np.zeros(order)
    P_hat = np.zeros((order, order))
    x_hat_apriori = np.zeros((order, 1))
    x_hat = np.zeros((order, 1))

    # Q = oe*np.eye(order) # This is incorrect but stable. Also should be oe**2 not oe

    # This is correct, but likely to be unstable
    Q = np.zeros((order, order))
    Q[0, 0] = oe**2

    a = get_autoreg_model(order, weights)

    h[0] = 1.0

    x_hat[:, 0] = y_signal[0 : order]
    P_hat[idx, idx] = p0

    store_x_hat = np.zeros((order, 1, num))
    store_P_hat = np.zeros((order, order, num))
    store_x_hat[:, :, order] = x_hat
    store_P_hat[:, :, order] = P_hat

    store_W = np.zeros((order, 1, num))
    store_S_Outer_W = np.zeros((order, order, num))
    store_Q = np.zeros((order, order, num))
    store_S = np.zeros((1, 1, num))

    predictions = np.zeros(n_testbefore + n_predict)


    # Start Filtering
    k = order # Wait until order number of msmts have been made
    while k < num:

        x_hat_apriori, P_hat_apriori = propagate_states_no_gamma(a, x_hat, P_hat, Q)

        if k > (n_train):
            # This loop is equivalent to setting the gain to zero (forecasting)
            x_hat = x_hat_apriori
            store_x_hat[:, :, k] = x_hat
            P_hat = P_hat_apriori
            store_P_hat[:, :, k] = P_hat
            k = k + 1
            continue

        W_, S = calc_Kalman_Gain(h, P_hat_apriori, rk, quantised=quantised, x_hat_apriori=x_hat_apriori) # W needs to be reshaped
        W = W_.reshape(order, 1)

        store_S[:, :, k] = S

        # Skip msmts
        if k % skip_msmts != 0:
            W = np.zeros((order, 1))

        e_z[k] = calc_residuals(h, x_hat_apriori, y_signal[k], quantised=quantised)

        inter = W*e_z[k]

        x_hat = x_hat_apriori + W*e_z[k]
        store_S_Outer_W[:, :, k] = S*np.outer(W, W.T)
        P_hat = P_hat_apriori - S*np.outer(W, W.T) # Equivalent to outer(W, W)

        store_x_hat[:, :, k] = x_hat
        store_P_hat[:, :, k] = P_hat
        store_W[:, :, k] = W

        k = k + 1

    if  save == 'Yes':

        np.savez(descriptor+'_AKF_', descriptor=descriptor+'_AKF_',
                 y_signal=y_signal,
                 order=order,
                 x_hat=store_x_hat,
                 P_hat=store_P_hat,
                 a=a,
                 h=h,
                 weights=weights,
                 e_z=e_z,
                 W=store_W,
                 Q=store_Q,
                 S=store_S,
                 oe=oe,
                 rk=rk,
                 n_train=n_train,
                 n_predict=n_predict,
                 n_testbefore=n_testbefore,
                 skip_msmts=skip_msmts)

    if quantised == 'No':
        return store_x_hat[0, 0, n_train - n_testbefore: ]
    else:
        return projected_msmt(store_x_hat[0, 0, n_train - n_testbefore: ])
