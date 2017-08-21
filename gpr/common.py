'''
PACKAGE: gpr
MODULE: gpr.common

The purpose of gpr is to implement Gaussian Process Regression (GPR) using GPy.
At present, the kernel being testing is the Periodic Kernel.

MODULE PURPOSE: Builds training data for GPR with Periodic Kernel in GPy.
METHODS:
get_data -- Returns data inputs in desired format for GPy implementation
    
'''
from __future__ import division, print_function, absolute_import

import numpy as np
import sys
sys.path.append('../')
from data_tools.load_raw_cluster_data import LoadExperiment as le
from data_tools.common import get_data as fetchdata

def add_axis(some_array):
    '''Adds empty new axis'''
    return some_array[:, np.newaxis]

def get_data(dataobject, points=200, 
             randomize='y'):
    '''
    Returns a set of data inputs in desired format for GPy implementation of 
    Gaussian Process Regression with a Periodic Kernel
        x -- Randomly chosen points on the time axis at which msmts are recieved
        y -- Randomly chosen corresponding msmts given by a true f(n)
        testx -- Deterministic set of test pts for n in [n_train - n_testbefore, n_train + n_predict]
        truth -- true f(n) taken from _BR_MAP.npz in LKFFB cluster data
        n_predict -- number of points in forecasting region, i.e. n_predict + n_train = number_of_points

    Using...
        test_case, variation -- specifies scenario under consideration for data_object
        pathtodir -- specifies path to LKFFB cluster data (_BR_AKF_MAP.npz) with 75*50 possible realisations of true f(n)
    '''

    # dataobject = le(test_case, variation, 
    #                 skip = 1,
    #                 GPRP_load='No', GPRP_path = './',
    #                 LKFFB_load = 'Yes', LKFFB_path = pathtodir,
    #                 AKF_load='No', AKF_path = './',
    #                 LSF_load = 'No', LSF_path = './')

    msmts, idx_truth = fetchdata(dataobject)

    num = dataobject.Expt.number_of_points
    n_predict = dataobject.Expt.n_predict
    n_train = dataobject.Expt.n_train
    n_testbefore = dataobject.Expt.n_testbefore

    shape = dataobject.LKFFB_macro_truth.shape
    macro_truth = dataobject.LKFFB_macro_truth.reshape(shape[0]*shape[1], shape[2]) # collapse  first two axees (only relevant to KF techniques)
    truth = macro_truth[idx_truth, :]

    timeaxis = np.arange(0, num, 1.0)

    if randomize =='y':
        x =[]
        y =[]

        for index in np.random.uniform(low=0, high=n_train, size=points):
            x.append(timeaxis[index])
            y.append(msmts[index])
    
    elif randomize != 'y':
        x = np.arange(0, n_train, 1.0, dtype=np.float32)
        y = msmts[0:n_train]

    testx = np.arange(n_train-n_testbefore, num, 1.0, dtype=np.float32)

    X = add_axis(np.asarray(x,dtype=np.float32)[0:n_train])
    Y = add_axis(np.asarray(y,dtype=np.float32)[0:n_train])
    TestX = add_axis(testx)

    
    return X, Y, TestX, truth, msmts
    
    
def simple_unlearnable_sine(nt=2000, delta_t = 0.001, f0 = 10., testpts=50, randomise='y'):
    
    print("Fourier resolution at training", 1.0/(nt*delta_t))
    print("True Frequency is", f0/3.)
    
    timeaxis = np.arange(0, nt+testpts, 1.0)
    truth = np.sin(2.0*np.pi*f0*(1./3)*delta_t*timeaxis)
    msmts = np.sin(2.0*np.pi*f0*(1./3)*delta_t*timeaxis) # no noise
    
    if randomise =='y':
        x =[]
        y =[]

        for index in np.random.uniform(low=0, high=nt, size=nt): # n_train == no of random msmts
            x.append(timeaxis[index])
            y.append(msmts[index])
    
    elif randomise != 'y':
        x = np.arange(0, nt, 1.0, dtype=np.float32)
        y = msmts[0:nt]

    testx = np.arange(nt-50, nt+testpts, 1.0, dtype=np.float32)

    X = add_axis(np.asarray(x,dtype=np.float32)[0:nt])
    Y = add_axis(np.asarray(y,dtype=np.float32)[0:nt])
    TestX = add_axis(testx)

    
    return X, Y, TestX, truth, msmts
