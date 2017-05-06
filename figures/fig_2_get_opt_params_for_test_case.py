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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

test_case=7
num_of_variations = 7

n_predict_ = 100
optimal_params_list =[]
msmt_noise_variance_list = []
fsize=13.5

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
    msmt_noise_variance_list.append(plotter_BR.msmt_noise_variance)
    
np.savez(os.path.join(savetopath_, str(filename0_)+'optimalparams.npz'), optimal_params_list=optimal_params_list, msmt_noise_variance_list=msmt_noise_variance_list)

sigma = np.array([x[0] for x in optimal_params_list])
R = np.array([x[1] for x in optimal_params_list])
msmt_noise_variance = np.array([x for x in msmt_noise_variance_list])


gs = gridspec.GridSpec(1,2, left=0.06, right=0.97, top=0.99, hspace=0.1,
                       wspace=0.25, bottom=0.1)
fig = plt.figure(figsize=(20,6))
ax_sigma = fig.add_subplot(gs[0, 1])
ax_sigma.set(ylabel=r'$\sigma$' , xlabel="Measurement Noise Variance")
ax_sigma.scatter(msmt_noise_variance, sigma)

ax_R = fig.add_subplot(gs[0, 0])
ax_R.scatter(msmt_noise_variance, R)
ax_R.set(ylabel=r'$R$' , xlabel="Measurement Noise Variance")
for ax in [ax_R, ax_sigma]:
    for item2 in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item2.set_fontsize(fsize)
    ax.tick_params(direction='in', which='both')

fig.savefig(savetopath_+'_sigmaR_v_msmtnoisevar.svg')
plt.close(fig)