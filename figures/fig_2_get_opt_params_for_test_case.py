#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Extracts optimal parameters from all variations in test_case_7, 8 or 9

@author: riddhisw
"""
import sys
sys.path.append('../')

import os
import numpy as np
from analysis_tools.plot_BR import Plot_BR_Results


test_case=7
num_of_variations = 7
n_predict_ = 100
optimal_params_list =[]

for var in range(1, num_of_variations + 1 , 1):
    variation = var
    savetopath_ = '/scratch/RDS-FSC-QCL_KF-RW/Kalman/test_case_'+str(test_case)+'/'
    filename0_ = 'test_case_'+str(test_case)+'_var_'+str(variation)
    brmapfile = filename0_+str('BR_Map')
    filename_and_path_BR = os.path.join(savetopath_, str(brmapfile)+'.npz')
    plotter_BR = Plot_BR_Results(filename_and_path_BR)

    filename_and_path_BR = os.path.join(savetopath_, str(brmapfile)+'.npz')
    plotter_BR = Plot_BR_Results(filename_and_path_BR)
    plotter_BR.load_data()
    plotter_BR.get_tuned_params(n_predict_)
    plotter_BR.make_plot()
    print("Optimal params stored = ", plotter_BR.lowest_pred_BR_pair)

    optimal_params_list.append(plotter_BR.lowest_pred_BR_pair)
    
np.savez(os.path.join(savetopath_, str(filename0_)+'optimalparams.npz'), optimal_params_list=optimal_params_list)