# The purpose of this script is to plot the optimisation procedure for a 
# a particular variation within a test case. 

import os
from analysis_tools.plot_BR import Plot_BR_Results
from analysis_tools.common import truncate_losses_
import matplotlib.pyplot as plt 

test_case=8
variation=7
undersamp_strength=562
savefig='Yes'

max_forecast_loss=50
fsize=12

savetopath_ = '/scratch/RDS-FSC-QCL_KF-RW/Kalman/test_case_'+str(test_case)+'/'
filename0_ = 'test_case_'+str(test_case)+'_var_'+str(variation)
brmapfile = filename0_+str('BR_Map')
    
filename_and_path_BR = os.path.join(savetopath_, str(brmapfile)+'.npz')
plotter_BR = Plot_BR_Results(filename_and_path_BR)

plotter_BR.load_data()
plotter_BR.get_tuned_params(max_forecast_loss)

for means_ind in xrange(2):
    vars()['x_data_tmp'+str(means_ind)], vars()['y_data_tmp'+str(means_ind)] = truncate_losses_(plotter_BR.means_lists_[means_ind], plotter_BR.truncation)

R = [x[1] for x in plotter_BR.random_hyperparams_list]
sigma = [x[0] for x in plotter_BR.random_hyperparams_list]

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))

ax.set_xscale('log')
ax.set_yscale('log')
for index in vars()['x_data_tmp'+str(0)]:
    ax.plot(sigma[index], R[index], 'ro', markersize=40, alpha=0.6)
for index in vars()['x_data_tmp'+str(1)]:
    ax.plot(sigma[index], R[index], 'go', markersize=25, alpha=0.6)
ax.plot(sigma, R, 'ko', markersize=10, label='Test Points')
ax.plot(plotter_BR.lowest_pred_BR_pair[0], plotter_BR.lowest_pred_BR_pair[1], 'x', color='yellow',  markersize=25, mew=3, label='Lowest Prediction Loss')
ax.set_xlabel(r' Process Noise Design Parameter $\sigma $ [Units: signal$^2$]')
ax.set_ylabel(r' Msmt Noise Design Parameter $R$ [Units: signal$^2$]')
ax.set_xlim([10**-11,1000])
ax.set_ylim([10**-11,1000])
ax.set_title(r'Prediction & Forecasting Bayes Risk Overlap v. Kalman Design Parameters ($\sigma, R$)')
for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fsize)
            
fig.subplots_adjust(left=0.1, right=0.95, wspace=0.25, hspace=0.25, top=0.8)
if savefig=='Yes':
    fig.savefig(savetopath_+filename0_+'loss_map.svg', format="svg")
