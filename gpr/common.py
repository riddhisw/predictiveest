'''
.. module:: gpr.common

    :synopsis: Build training data for GPR with Periodic Kernel in GPy.

    Module Level Functions:
    ----------------------
        get_data : Return data inputs in desired format for GPy implementation.
        add_axis : Add an empty new axis. [Helper Function].

.. moduleauthor:: Riddhi Gupta <riddhi.sw@gmail.com>

'''
from __future__ import division, print_function, absolute_import

import numpy as np
import sys
sys.path.append('../')
from data_tools.load_raw_cluster_data import LoadExperiment as le
from data_tools.common import get_data as fetchdata

def add_axis(some_array):
    '''Adds empty new axis. [Helper Function].'''
    return some_array[:, np.newaxis]


def get_data(dataobject, points=200, randomize='y'):
    '''Return a set of data inputs in desired format for GPy implementation of
       Gaussian Process Regression with a Periodic Kernel.

       Parameters:
       ----------
            dataobject (`class object`) :  A data_tools.load_raw_cluster_data.LoadExperiment instance.
            randomize (`str`, optional): A Yes (`y`) / No (`n`) flag to randomize choice of time labels.
            points (`int`) : Number of time labels chosen for GPR analysis.

       Returns:
       -------
            x (`float64`): Randomly chosen points on the time axis at which msmts
                are recieved.
            y (`float64`): Randomly chosen corresponding msmts given by a
                true f(n).
            testx (`float64`): Deterministic set of test pts for n in
                [n_train - n_testbefore, n_train + n_predict].
            truth (`float64`): true f(n) taken from _BR_MAP.npz in LKFFB cluster data.
            n_predict (`int`): number of points in forecasting region,
                i.e. n_predict + n_train = number_of_points.
    '''
    msmts, idx_truth = fetchdata(dataobject)

    num = dataobject.Expt.number_of_points
    n_predict = dataobject.Expt.n_predict
    n_train = dataobject.Expt.n_train
    n_testbefore = dataobject.Expt.n_testbefore

    shape = dataobject.LKFFB_macro_truth.shape

    # Collapse  first two axees (only relevant to KF techniques)
    macro_truth = dataobject.LKFFB_macro_truth.reshape(shape[0]*shape[1], shape[2])
    truth = macro_truth[idx_truth, :]

    timeaxis = np.arange(0, num, 1.0)

    if randomize == 'y':
        x = []
        y = []

        for index in np.random.uniform(low=0, high=n_train, size=points):
            x.append(timeaxis[index])
            y.append(msmts[index])

    elif randomize != 'y':
        x = np.arange(0, n_train, 1.0, dtype=np.float32)
        y = msmts[0:n_train]

    testx = np.arange(n_train-n_testbefore, num, 1.0, dtype=np.float32)

    # GPy format
    X = add_axis(np.asarray(x,dtype=np.float32)[0:n_train])
    Y = add_axis(np.asarray(y,dtype=np.float32)[0:n_train])
    TestX = add_axis(testx)

    return X, Y, TestX, truth, msmts


def simple_unlearnable_sine(nt=2000, delta_t=0.001, f0=10., testpts=50, randomise='y'):
    '''Helper Function [DEPRECIATED]'''

    # print("Fourier resolution at training", 1.0/(nt*delta_t))
    # print("True Frequency is", f0/3.)

    timeaxis = np.arange(0, nt+testpts, 1.0)
    truth = np.sin(2.0*np.pi*f0*(1./3)*delta_t*timeaxis)
    msmts = np.sin(2.0*np.pi*f0*(1./3)*delta_t*timeaxis) # no noise

    if randomise == 'y':
        x =[]
        y =[]

        for index in np.random.uniform(low=0, high=nt, size=nt): # n_train == no of random msmts
            x.append(timeaxis[index])
            y.append(msmts[index])

    elif randomise != 'y':
        x = np.arange(0, nt, 1.0, dtype=np.float32)
        y = msmts[0:nt]

    testx = np.arange(nt-50, nt+testpts, 1.0, dtype=np.float32)

    X = add_axis(np.asarray(x, dtype=np.float32)[0:nt])
    Y = add_axis(np.asarray(y, dtype=np.float32)[0:nt])
    TestX = add_axis(testx)

    return X, Y, TestX, truth, msmts
