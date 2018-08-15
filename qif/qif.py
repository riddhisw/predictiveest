'''
Created on Thu Apr 20 19:20:43 2017
@author: riddhisw

.. module:: qif.qif

    :synopsis: Converts an autoregressive process order (q) in a state space model,
        and implements a Quantised Kalman Filter (QKF).

    Module Level Functions:
    ----------------------
        qif : Saves results of Quantised Kalman Filtering (QKF) with AR dynamics as .npz file.

.. moduleauthor:: Riddhi Gupta <riddhi.sw@gmail.com>
'''

from __future__ import division, print_function, absolute_import

import numpy as np
import sys
sys.path.append('../')

from akf.armakf import get_autoreg_model
from qif.common import calc_residuals, calc_gain, projected_msmt, propagate_x, propagate_p, update_p, calc_z_proj


def qif(descriptor, y_signal, weights, oe, rk, n_train=1000, n_testbefore=50,
        n_predict=50, p0=10000, skip_msmts=1,  save='No'):

    '''
    Saves results of Quantised Kalman Filtering (QKF) with AR dynamics as .npz file.

    Parameters:
    ----------
        descriptor (`str`): Descriptive filename for saving QKF output as .npz file.
        y_signal (`float64`) : Time stamped measurement sequence [Dim: 1 x num ].
        weights (`float64`) : Coefficients of an AR(q) process  [Dim: 1 x order].
        oe (`float64`) : Kalman process noise variance parameter [Dim: 1 x 1].
        rk (`float64`) : Kalman measurement noise variance parameter [Dim: 1 x 1].
        n_train (`int`, optional) : Number of measurements during state estimation [Dim: 1 x 1].
            Defaults to 1000.
        n_testbefore (`int`, optional) : Number of time-steps before training ceases [Dim: 1 x 1].
            Defaults to 50.
        n_predict (`int`, optional) : Number of time-steps after training ceases [Dim: 1 x 1].
            Defaults to 50.
        p0 (`float64`, optional) : Initial state variance [Dim: 1 x 1].
            Defaults to 10000.
        skip_msmts=1 (`str`, optional): Number of measurements - 1 to skip.
            Defaults to 1.
        save=(`str`, optional): 'Yes' / 'No' flag. Saves QKF output as .npz file if 'Yes'.
            Defaults to 'No'.

    Returns:
    -------
        Saves results of Kalman Filtering with AR dynamics as .npz file.
    '''

    num = y_signal.shape[0]
    order = weights.shape[0]

    e_z = np.zeros(num)

    idx = range(order)
    P_hat = np.zeros((order, order))
    x_hat_apriori = np.zeros((order, 1))
    x_hat = np.zeros((order, 1))


    # # QKF Noise Covariance Matrix
    # This is stable form for the covariance matrix but not standard for AR processes.
    # We use oe**2 so that our tuning is done on
    # oe (st dev) and rk (variance) as consistent with AKF, LKFFB.
    Q = (oe)*np.eye(order)

    # The form of Q below is technically correct, but  unstable numerically.
    # Q = np.zeros((order, order))
    # # This is used in AKF. Makes marginal difference to test cases in Fig 8a,
    # but prohibits numerics e.g. recursion for Fisher information as Q^{-1} doesn't exist.
    # Q[0,0] = oe # # This is used in AKF with Q[0,0] = oe**2

    a = get_autoreg_model(order, weights)
    x_hat[:, 0] = np.random.normal(scale=np.sqrt(oe), size=order) # y_signal[0:order]
    P_hat[idx, idx] = p0

    store_x_hat = np.zeros((order, 1, num))
    store_P_hat = np.zeros((order, order, num))
    store_x_hat[:, :, order] = x_hat
    store_P_hat[:, :, order] = P_hat

 
    store_W = np.zeros((order, 1, num))
    store_S_Outer_W = np.zeros((order, order, num))
    store_Q = np.zeros((order, order, num))
    store_S = np.zeros((1, 1, num))

    predictions = np.zeros(num)


    # Start Filtering
    k = order # wait until order number of msmts have been made
    while (k< num):
        # print k
        x_hat_apriori = propagate_x(a, x_hat)
        P_hat_apriori = propagate_p(a, P_hat, Q)

        # Make predictions
        z_proj = calc_z_proj(x_hat_apriori) # potentially non linear msmt h(x)
        predictions[k] = projected_msmt(z_proj) # quantisation

        # Residuals / innovations
        e_z[k] = calc_residuals(predictions[k], y_signal[k])

        # Zero Gain
        if k > (n_train):
            # This loop is equivalent to setting the gain to zero (forecasting)
            x_hat = x_hat_apriori
            store_x_hat[:, :, k] = x_hat
            P_hat = P_hat_apriori
            store_P_hat[:, :, k] = P_hat
            k = k + 1
            continue

        # Non Zero Gain
        W_, S = calc_gain(x_hat_apriori, P_hat_apriori, rk)  # W needs to be reshaped
        W = W_.reshape(order, 1)

        store_S[:, :, k] = S

        # Skip msmts
        if k % skip_msmts != 0:
            W = np.zeros((order, 1))

        # Kalman Update
        x_hat = x_hat_apriori + W*e_z[k]
        P_hat = update_p(P_hat_apriori, S, W)

        store_x_hat[:, :, k] = x_hat
        store_P_hat[:, :, k] = P_hat
        store_W[:, :, k] = W

        k = k + 1

    if  save == 'Yes':

        np.savez(descriptor+'_QIF_', descriptor=descriptor+'_QIF_',
                 y_signal=y_signal,
                 order=order,
                 x_hat=store_x_hat,
                 P_hat=store_P_hat,
                 predictions=predictions,
                 a=a,
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

    return predictions, store_W, store_x_hat, store_P_hat, e_z


