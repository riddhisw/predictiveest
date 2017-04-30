#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Uses optimal parameters from all variations in test_case_8 to run KF on a single truth + dataset

@author: riddhisw
"""

import os
import numpy as np
from kf.fast import kf_2017 as kf
import matplotlib.pyplot as plt

test_case=7
n_train_ = [0, 2000, 1000, 667, 500, 400, 200, 125]
n_predict_ = [0, 100, 50, 33, 25, 20, 10, 7]
n_testbefore_ = [0, 50, 25, 17, 13, 10, 5, 3 ]
multiplier_ = [0, 20, 10, 6.66666666667, 5, 4, 2, 1.25]
bandwidth_ = 50.0
skip_msmts = [0, 1, 2, 3, 4, 5, 10, 16]
optimal_params_list =[]
x0 = 1.0
p0 = 100000.0
freq_basis_array = np.arange(0.0, 50.0, 0.5)

savetopath_ = '/scratch/RDS-FSC-QCL_KF-RW/Kalman/test_case_'+str(test_case)+'/'
truth = np.load(os.path.join(savetopath_,'test_case_'+str(test_case)+'_var_1_kfresults_skip_msmts_Truth.npz' ))['truth']
data = np.load(os.path.join(savetopath_,'test_case_'+str(test_case)+'_var_1_kfresults_skip_msmts_Truth.npz' ))['noisydata']
optimal_params_list = np.load(os.path.join(savetopath_,'test_case_'+str(test_case)+'_var_7optimalparams.npz' ))['optimal_params_list']

predictions_list =[]

plt.figure(figsize=(15,8))

for var in [1, 2, 5, 6]:
    variation = var
    Delta_T_Sampling = 1.0/(multiplier_[var]*bandwidth_)
    Time_Axis = Delta_T_Sampling*np.arange(-n_testbefore_[var], n_predict_[var], 1)
    
    print("Delta T Sampling =", Delta_T_Sampling)
    
    oe, rk = optimal_params_list[var-1, :]
        
    if var==7:
        delT0 = 1.0/(multiplier_[1]*bandwidth_)
        prediction2 = kf(data, n_train_[1], n_testbefore_[1], n_predict_[1], 
                        delT0, x0, p0, oe, rk, freq_basis_array, phase_correction=0, 
                        prediction_method="ZeroGain", skip_msmts=skip_msmts[var], descriptor='Fast_KF_Results')
        plt.plot(delT0*np.arange(-n_testbefore_[1], n_predict_[1], skip_msmts[1]), prediction2, 'o', label='Skip Msmts (via Zero Gain) =%s'%(skip_msmts[var]))
        
    predictions = kf(data[::skip_msmts[var]], n_train_[var], n_testbefore_[var], n_predict_[var], 
                    Delta_T_Sampling, x0, p0, oe, rk, freq_basis_array, phase_correction=0, 
                    prediction_method="ZeroGain", skip_msmts=1, descriptor='Fast_KF_Results')
    predictions_list.append(predictions)
    
    plt.plot(Time_Axis, predictions, 'x-', label='Skip Msmts (via Sampling Rates) =%s'%(skip_msmts[var]))
plt.plot((1.0/(multiplier_[1]*bandwidth_))*np.arange( -n_testbefore_[1], n_predict_[1], 1), truth[n_train_[1] - n_testbefore_[1] : n_train_[1] + n_predict_[1]], 'k-', label='Truth')
plt.legend(loc=0)
plt.show()