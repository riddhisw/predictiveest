import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from analysis_tools.case_data_explorer import CaseExplorer as cs

########################################
# INPUT DATA 
########################################

path_to_directory = '/scratch/RDS-FSC-QCL_KF-RW/Kalman'
savefigname = 'tc_7_'
ADD_LS_DATA = 'Yes'
DO_SKF = 'No'

test_case_list = [7, 7, 7, 7, 7] # Equal len
variation_list = [1, 2, 4, 6, 7] # Equal len

max_stp_fwd=[]

story1 = [2.0, 1.0, 0.998, 0.994, 0.99] # r'$f_0 / \Delta\omega^B $'
story2 = [0.1988, 0.3976, 0.7952, 1.1928, 1.988] # r'$f_0 J / \Delta\omega^B_{max} $'
story3 = [1.8181818182, 0.9523809524, 0.487804878, 0.3278688525, 0.2469135802] # r'$ \Delta s$ '
story4 = [0.01, 0.05, 0.1, 0.2, 0.25] # r'Msmt Noise Lvl [% of 3 St. Dev $f_n$ Pts]'
story6 = [0.1777777778, 0.3555555556, 0.8000000001, 1.0666666668, 1.4222222224] # r'$f_0 J / \Delta\omega^B_{max} $'
story7 = [0.9090909091, 0.4761904762, 0.3225806452, 0.243902439, 0.1960784314] #r'$ \Delta s$ '
story8 = story4
tc7 = [20, 10, 5, 2, 1.25] # 'Nyquist r [dimless]'
tc8 = tc7
tc10 = tc7
tc12 = tc7
tc13 = tc7 
tc14 = tc7

dial = tc7
dial_label = 'Nyquist r [dimless]'
# r'$\Delta \omega^B / f_0$'

loss_hist_min = 10**-2
loss_hist_max = 10**6
amp_PSD_min = 10**-5
stps_fwd_truncate_=50
kea_max = 10**3

## Scenarios with Changing Delta T
n_predict_list = [0, 100, 50, 33, 25, 20, 10, 7] 
n_testbefore_list = [0, 50, 25, 17, 13, 10, 5, 3 ] 

## Scenarios with no Changing Delta T
# n_testbefore_list = [0, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
# n_predict_list = n_testbefore_list 
# n_predict_list = [0, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100]


########################################
# REFERENCE PARAMETERS (NO CHANGE) 
########################################

max_forecast_loss_list = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
skip_list = [1, 1, 1 , 1 , 1 , 1, 1, 1, 1, 1, 1, 1, 1]
skip_list_2 = [0, 1, 2, 3, 4, 5, 10, 16]

NUM_SCENARIOS = len(test_case_list) # == len(variation_list)

Hard_load='No' 
SKF_load='No'


for idx in xrange(NUM_SCENARIOS):
        vars()['obj_'+str(test_case_list[idx])+'_'+str(variation_list[idx])] = cs(
            test_case_list[idx],
            variation_list[idx],
            skip_list[idx],
            max_forecast_loss_list[idx],
            path_to_directory)

## Amplitudes
FUDGE = 0.5
HILBERT_TRANSFORM = 2.0

## Kalman Basis
BASIS_PRED_NUM = 0 # or 1 for Basis A

########################################
# FIG: Grid Specs
########################################
ROWS = 4
gs = gridspec.GridSpec(ROWS, NUM_SCENARIOS ,
                       left=0.08, right=0.97, 
                       top=0.9, bottom=0.05, 
                       wspace=0.6, hspace=0.5)

fig = plt.figure(figsize=( 3.0*(NUM_SCENARIOS ), 3.0*4))

count=0

for idx in xrange((NUM_SCENARIOS)):
    for idx_ax2 in xrange(ROWS):
        if variation_list[idx]==7 and DO_SKF=='Yes':
            vars()['ax_var'+str(variation_list[idx])+'_'+str(idx_ax2)] = fig.add_subplot(gs[idx_ax2, idx], facecolor='mistyrose')
        else:
            vars()['ax_var'+str(variation_list[idx])+'_'+str(idx_ax2)] = fig.add_subplot(gs[idx_ax2, idx])
        #vars()['ax_var'+str(variation_list[idx])+'_'+str(idx_ax2)].locator_params(axis='x', numticks=4)

########################################
# FIG: Custom Legends
########################################
optimal_star = 'magenta'

l_train = mpatches.Patch(color='gray', alpha=0.3)
predictzeroline =  mlines.Line2D([], [], linestyle='-', color='darkblue')

randinit = mlines.Line2D([], [], linestyle='None', color=None, marker='v', markerfacecolor='k', markeredgecolor='k', markersize=7)

optimalstar = mlines.Line2D([], [], linestyle='None',  color=None, marker='*', markerfacecolor=optimal_star, markeredgecolor=optimal_star, markersize=7, alpha=1)

pred_circ = mlines.Line2D([], [], linestyle='-',  color='tan', marker='o', markerfacecolor='tan', markeredgecolor='tan', markersize=7)
pred_line = mlines.Line2D([], [], linestyle='-',  color=optimal_star)

fore_circ = mlines.Line2D([], [], linestyle='-',  color='c', marker='o', markerfacecolor='c', markeredgecolor='c', markersize=7)
fore_line = mlines.Line2D([], [], linestyle='-',  color='teal')

un_opt_traj = mlines.Line2D([], [], linestyle='-', color='gray')
opt_traj = mlines.Line2D([], [], linestyle='-', color=optimal_star)


vars()['ax_var'+str(variation_list[2])+'_'+str(0)].legend(handles=(randinit,  un_opt_traj,  pred_circ, fore_circ, optimalstar, opt_traj),
                                                                  labels= [r'Random Init. ($\sigma, R$)', 'Unoptimal Risk Traj.', 'Low State Est. Risk', 'Low Prediction Risk', r'Tuned ($\sigma, R$)',  'Optimal Risk Traj.'],
                                                                  bbox_to_anchor=(-3.8, 1.21, 5.0, 0.2), loc=2, ncol=6, frameon=True, fontsize=12.5,
                                                                  facecolor='linen',
                                                                  edgecolor='white')


########################################
# FIG: Size, Colors and Labels
########################################
fsize=13.5
PLOT_SCALE = 1000
savefig='Yes'
us_colour_list = ['g', 'dodgerblue', 'purple', 'maroon', 'darkorange']

loss_color_list = ['tan', 'c', optimal_star, optimal_star]
style = ['-', '-', '-', '-']
ax_kea_labels=['A', 'B', 'C', 'D', 'E']
ax_tui_labels=['A*', 'B*', 'C*', 'D*', 'E*']

########################################
# FIG: Loss Map on Variational Plots
########################################

idx_ax2_list = [0, 2] # Loss Map on 1st and 3rd rows

for idx in xrange(NUM_SCENARIOS):

    obj_ = 'obj_'+str(test_case_list[idx])+'_'+str(variation_list[idx])
    output = vars()[obj_].return_low_loss_hyperparams_list() # this has data from kf and akf

    for idx_kf_type in xrange(2): 
        
        idx_ax2 = idx_ax2_list[idx_kf_type]

        s0, R0, s1, R1, sigma, R, p_index, p_losses = [item for item in output[idx_kf_type]]
        
        ax_ = vars()['ax_var'+str(variation_list[idx])+'_'+str(idx_ax2)]

        if idx==0:
            ax_.set_ylabel(r' Kalman $R$ [$f_n^2$]')
        if idx==2:
            ax_.set_xlabel(r' Kalman $\sigma $ [$x^2$]')
        
        ax_.set_xlim([10**-11,1000])
        ax_.set_ylim([10**-11,1000])
        ax_.set_xscale('log')
        ax_.set_yscale('log')
        ax_.plot(s0, R0, 'o', c='tan', markersize=25, alpha=0.7)
        ax_.plot(s1, R1, 'o', c='cyan', markersize=15, alpha=0.7)
        ax_.plot(sigma, R, 'kv', markersize=5, alpha=1.0)
        ax_.plot(s0[0], R0[0],'*', color=optimal_star, markersize=15, mew=2)
        ax_.tick_params(direction='in', which='both')

        if idx_ax2==0 :
            ax_.annotate(ax_kea_labels[idx], xy=(0, 1.5), 
                    xycoords=('axes fraction', 'axes fraction'),
                    xytext=(1,1),
                    textcoords='offset points',
                    size=24,
                    color=us_colour_list[idx],
                    ha='left',
                    va='center')

        if idx_ax2 == 0 and idx==0 :     
            ax_.annotate('KF - Basis of Oscillators', xy=(-0.5, 1.06), 
                    xycoords=('axes fraction', 'axes fraction'),
                    xytext=(1,1),
                    textcoords='offset points',
                    size=fsize*1.05,
                    color='r',
                    ha='left',
                    va='center')
        
        if idx_ax2 == 2 and idx==0:     
            ax_.annotate('KF - Autoregressive AR(q=101) with LS Weights', xy=(-0.5, 1.08), 
                    xycoords=('axes fraction', 'axes fraction'),
                    xytext=(1,1),
                    textcoords='offset points',
                    size=fsize*1.05,
                    color='r',
                    ha='left',
                    va='center')

########################################
# FIG: Loss Histograms on Variational Plots
########################################

idx_ax2_list = [1 , 3] # Loss Histogram on second and last rows

for idx in xrange(NUM_SCENARIOS):

    obj_ = 'obj_'+str(test_case_list[idx])+'_'+str(variation_list[idx])
    output = vars()[obj_].return_low_loss_hyperparams_list() # this has data from kf and akf

    for idx_kf_type in xrange(2): 
        
        idx_ax2 = idx_ax2_list[idx_kf_type]

        s0, R0, s1, R1, sigma, R, p_index, p_losses = [item for item in output[idx_kf_type]]

        ax_ = vars()['ax_var'+str(variation_list[idx])+'_'+str(idx_ax2)]

        start_at = vars()[obj_].n_testbefore - n_testbefore_list[variation_list[idx]] #== 0 if Delta_T is not changing
        end_at = n_predict_list[variation_list[idx]] #==  n_predict if Delta_T is not changing
        x_axis = PLOT_SCALE*vars()[obj_].Delta_T_Sampling*np.arange(start_at - vars()[obj_].n_testbefore, end_at, 1)
        opt_index = p_index[0]
        unopt_index_list = list(set(range(0, vars()[obj_].num_randparams, 1)) - set(p_index))

        count=0
        alpha=0.1
        for chosen_index in unopt_index_list:
            ax_.plot(x_axis[:n_testbefore_list[variation_list[idx]]], np.mean(vars()[obj_].macro_prediction_errors[chosen_index,:,start_at:], axis=0), 
                    '-', markersize=4, alpha=alpha, c='k')
            ax_.plot(x_axis[n_testbefore_list[variation_list[idx]] :], np.mean(vars()[obj_].macro_forecastng_errors[chosen_index,:,0:end_at], axis=0), 
                    '-', markersize=4, alpha=alpha, c='k')
        alpha=0.5
        count=0
        for chosen_index in p_index[1:] + [opt_index]:
            if chosen_index==opt_index:
                count=2
                alpha=1.0
            ax_.plot(x_axis[:n_testbefore_list[variation_list[idx]]], np.mean(vars()[obj_].macro_prediction_errors[chosen_index,:,start_at:], axis=0), 
                    style[count], markersize=4, alpha=alpha, c=loss_color_list[count])
            ax_.plot(x_axis[n_testbefore_list[variation_list[idx]] :], np.mean(vars()[obj_].macro_forecastng_errors[chosen_index,:,0:end_at], axis=0), 
                    style[count+1], markersize=4, alpha=alpha, c=loss_color_list[count+1])
        
        ax_.set_yscale('log')
        ax_.set_xlim([-50,50])
        ax_.set_ylim([loss_hist_min, loss_hist_max])
        ax_.axvspan(-50,0, color='gray', alpha=0.3)
        
        if idx==0:
            ax_.set_ylabel(r'log($\langle (f_n -\hat{f_n})^2 \rangle_D$) [log($f_n^2$)]')
        if idx==2:
            ax_.set_xlabel('Stps Fwd [num]')
        
        ax_.tick_params(direction='in', which='both')

######################################
# Font Sizes
######################################

for idx in xrange(NUM_SCENARIOS):
    for idx_ax2 in xrange(ROWS):
        ax = vars()['ax_var'+str(variation_list[idx])+'_'+str(idx_ax2)]
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fsize)
#        for item in ([ax.xaxis.label, ax.yaxis.label]):
#            item.set_weight('bold')

######################################
# Save and Close
######################################   
#    
fig.savefig(savefigname+'loss_comparison.png', format='png')
plt.close(fig)


