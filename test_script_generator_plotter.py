#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 15:47:10 2017

@author: riddhisw
"""
import os
from analysis_tools.plot_BR import Plot_BR_Results
from analysis_tools.plot_KF import Plot_KF_Results

test_case=1
variation=0
filepath = 'test_case_1/'
filename0_ = 'test_case_'+str(test_case)+'_var_'+str(variation)
savetopath_ = filepath
brmapfile = 'test_case_'+str(test_case)+'_var_'+str(variation)+str('BR_Map')
kfresultsfile= 'test_case_'+str(test_case)+'_var_'+str(variation)+'_kfresults'

n_train_ = 2000
n_predict_ = 50
n_testbefore_ = 50
multiplier_ = 20
bandwidth_ = 50.0

exp_params_ = [n_train_, n_predict_, n_testbefore_, multiplier_, bandwidth_]

filename_and_path_BR = os.path.join(savetopath_, str(brmapfile)+'.npz')
plotter_BR = Plot_BR_Results(filename_and_path_BR)
plotter_BR.make_plot()

for skip in [1, 2, 3, 4, 5, 10, 15]:
    
    filename_skippy = os.path.join(savetopath_, str(kfresultsfile)+'_skipmsmts_'+str(skip))
    plotter_KF = Plot_KF_Results(exp_params_, filename_skippy+'.npz')
    plotter_KF.make_plot()
    plotter_KF.show_one_prediction()
    plotter_KF.compare_skf_and_dkf(filename_skippy)

