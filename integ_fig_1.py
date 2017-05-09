import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from analysis_tools.case_data_explorer import CaseExplorer as cs


########################################
# DATA 
########################################

path_to_directory = '.'
savefigname = 'test_case_7_no'
test_case_list = [7, 7, 7, 7, 7] # Equal len
variation_list = [1, 2, 4, 6, 7] # Equal len
DO_SKF = 'No'

skip_list = [1, 1, 1 , 1 , 1 , 1 ]
NUM_SCENARIOS = len(test_case_list) # == len(variation_list)
max_forecast_loss_list=[50, 50, 50, 50, 50, 50]

Hard_load='No' 
SKF_load='No'


skip_list_2 = [0, 1, 2, 3, 4, 5, 10, 16]
for idx in xrange(NUM_SCENARIOS):
        vars()['obj_'+str(test_case_list[idx])+'_'+str(variation_list[idx])] = cs(
            test_case_list[idx],
            variation_list[idx],
            skip_list[idx],
            max_forecast_loss_list[idx],
            path_to_directory)

## All Amplitudes
FUDGE = 0.5
HILBERT_TRANSFORM = 2.0

## All KF EAs
BASIS_PRED_NUM = 0 # or 1 for Basis A

## All KF EAs with Changing Delta T
n_predict_list = [0, 100, 50, 33, 25, 20, 10, 7] # For plotting KF ensemble averages only
n_testbefore_list = [0, 50, 25, 17, 13, 10, 5, 3 ] # For plotting KF ensemble averages only
#skip_list=[0, 1, 2, 3,  4, 5, 10, 16] # For plotting KF ensemble averages only

########################################
# FIG: Grid Specs
########################################

gs = gridspec.GridSpec(4, NUM_SCENARIOS + 2,
                       left=0.05, right=0.97, 
                       top=0.95, bottom=0.05, 
                       wspace=0.75, hspace=0.65)

fig = plt.figure(figsize=( 3.0*(NUM_SCENARIOS + 2), 3.0*4))

count=0

ax_kea = fig.add_subplot(gs[0:2,0:2])

if DO_SKF == 'Yes':
    ax_tui = fig.add_subplot(gs[2:4,0:2])

for idx in xrange((NUM_SCENARIOS)):
    for idx_ax2 in xrange(4):
        vars()['ax_var'+str(variation_list[idx])+'_'+str(idx_ax2)] = fig.add_subplot(gs[idx_ax2, 2+idx])

########################################
# FIG: Custom Legends
########################################


########################################
# FIG: Size, Colors and Labels
########################################
fsize=13.5
PLOT_SCALE = 1000
savefig='Yes'
us_colour_list = [0, 'dodgerblue', 'deepskyblue', 'b', 'darkblue', 'purple', 'maroon', 'deeppink', 'saddlebrown', 'darkorange', 'olive', 'darkolivegreen', 'mediumseagreen', 'g', 'limegreen']
loss_color_list = ['crimson', 'crimson', 'tan', 'c']
ax_kea_labels=[]
ax_tui_labels=[]
########################################
# FIG: Plot Kalman Averages on First Plot
########################################

for idx in xrange(NUM_SCENARIOS):

    obj_ = 'obj_'+str(test_case_list[idx])+'_'+str(variation_list[idx])

    start_at = vars()[obj_].n_testbefore - n_testbefore_list[variation_list[idx]]
    end_at = n_predict_list[variation_list[idx]] + vars()[obj_].n_testbefore

    x_axis = PLOT_SCALE*vars()[obj_].Delta_T_Sampling*np.arange(start_at - vars()[obj_].n_testbefore, 
                                                                end_at - vars()[obj_].n_testbefore ,
                                                                1)

    ax_kea.plot(x_axis, vars()[obj_].Normalised_Means[BASIS_PRED_NUM,start_at :end_at], '--', c=us_colour_list[variation_list[idx]])
    ax_kea.set(xlabel='Stps Fwd [num]', ylabel=r'$Norm. log(E(err^2)$ [log($f_n^2$)]')
    ax_kea.set_yscale('log')
    ax_kea.set_ylim([10**(-5), 5])
    ax_kea_labels.append('Some label')
    ax_kea.tick_params(direction='in', which='both')

ax_kea.axvspan(-50,0, color='gray', alpha=0.3)
ax_kea.axhline(1.0, linestyle='-', color='darkblue')

########################################
# FIG: Loss Map on Variational Plots
########################################

idx_ax2 = 0# Loss Map

for idx in xrange(NUM_SCENARIOS):

    obj_ = 'obj_'+str(test_case_list[idx])+'_'+str(variation_list[idx])
    s0, R0, s1, R1, sigma, R, p_index, p_losses = vars()[obj_].return_low_loss_hyperparams_list()
    
    ax_ = vars()['ax_var'+str(variation_list[idx])+'_'+str(idx_ax2)]
    ax_.set_xlabel(r' Kalman $\sigma $ [$x^2$]')
    ax_.set_ylabel(r' Kalman $R$ [$f_n^2$]')
    ax_.set_xlim([10**-11,1000])
    ax_.set_ylim([10**-11,1000])
    ax_.set_xscale('log')
    ax_.set_yscale('log')
    ax_.plot(s0, R0, 'o', c='tan', markersize=25, alpha=0.7)
    ax_.plot(s1, R1, 'o', c='cyan', markersize=15, alpha=0.7)
    ax_.plot(sigma, R, 'kv', markersize=5, alpha=1.0)
    ax_.plot(s0[0], R0[0],'*', color='crimson', markersize=15, mew=2)
    ax_.tick_params(direction='in', which='both')


########################################
# FIG: Loss Histograms on Variational Plots
########################################

idx_ax2 = 1 # Loss Histogram

for idx in xrange(NUM_SCENARIOS):

    obj_ = 'obj_'+str(test_case_list[idx])+'_'+str(variation_list[idx])
    s0, R0, s1, R1, sigma, R, p_index, p_losses = vars()[obj_].return_low_loss_hyperparams_list()

    ax_ = vars()['ax_var'+str(variation_list[idx])+'_'+str(idx_ax2)]

    start_at = vars()[obj_].n_testbefore - n_testbefore_list[variation_list[idx]] #== 0 if Delta_T is not changing
    end_at = n_predict_list[variation_list[idx]] #==  n_predict if Delta_T is not changing
    x_axis = PLOT_SCALE*vars()[obj_].Delta_T_Sampling*np.arange(start_at - vars()[obj_].n_testbefore, end_at, 1)
    opt_index = p_index[0]
    unopt_index = list(set(range(0, vars()[obj_].num_randparams, 1)) - set(p_index))[0] #just choosing the first one

    count=0
    for chosen_index in [opt_index, unopt_index]:
        ax_.plot(x_axis[:n_testbefore_list[variation_list[idx]]], np.mean(vars()[obj_].macro_prediction_errors[chosen_index,:,start_at:], axis=0), 'o-', markersize=4, c=loss_color_list[count])
        count+=1
        ax_.plot(x_axis[n_testbefore_list[variation_list[idx]] :], np.mean(vars()[obj_].macro_forecastng_errors[chosen_index,:,0:end_at], axis=0), 'o-', markersize=4, c=loss_color_list[count])
        count+=1
    
    ax_.set_yscale('log')
    ax_.set_xlim([-50,100])
    ax_.set_ylim([10**-1, 10**5])
    ax_.axvspan(-50,0, color='gray', alpha=0.3)
    ax_.set(xlabel='Stps Fwd [num]', ylabel=r'$ Un. Norm log(E(err^2)$ [log($f_n^2$)]')
    ax_.tick_params(direction='in', which='both')

########################################
# FIG: Amplitude Graphs for Variation Plots
########################################
idx_ax2 = 2 # Amplitudes from Single Run

for idx in xrange(NUM_SCENARIOS):

    obj_ = 'obj_'+str(test_case_list[idx])+'_'+str(variation_list[idx])
    x_data, y_data, true_S_norm = vars()[obj_].return_fourier_amps()

    ax = vars()['ax_var'+str(variation_list[idx])+'_'+str(idx_ax2)]

    ax.set(xlabel=r'$\omega$ [rad]', ylabel=r'$S(\omega)$ [$f_n^2$/(rad $s^{-1}$)]')
    ax.plot(x_data[0], y_data[0], 'o', c=us_colour_list[variation_list[idx]])
    ax.ticklabel_format(style='sci', scilimits=(0,2), axis='y')
    for label in ax.get_yticklabels():
        label.set_fontsize(fsize)
        label.set_color(us_colour_list[variation_list[idx]])

    ax.annotate('T.Pow: %.2e'%(np.round(np.sum(y_data[0]))), xy=(0.8,1.03), 
                xycoords=('axes fraction', 'axes fraction'),
                xytext=(1,1),
                textcoords='offset points',
                size=10,
                color=us_colour_list[variation_list[idx]],
                ha='right',
                va='center')
    # Theory

    ax2 = ax.twinx()
    ax2.tick_params(direction='in', which='both')

    ax2.plot(x_data[1], y_data[1], 'r')
    ax2.ticklabel_format(style='sci', scilimits=(0,2), axis='y')
    
    for label in ax2.get_yticklabels():
        label.set_fontsize(fsize)
        label.set_color('r')

    ax2.annotate('T.Pow: %.2e'%(np.round(true_S_norm)), xy=(0.8,1.11), 
                xycoords=('axes fraction', 'axes fraction'),
                xytext=(1,1),
                textcoords='offset points',
                size=10,
                color='r',
                ha='right',
                va='center')

########################################
# FIG: Single Prediction in Natural Timescale 
########################################
idx_ax2 = 3 # Single Prediction in Natural Timescale


for idx in xrange(NUM_SCENARIOS):

    obj_ = 'obj_'+str(test_case_list[idx])+'_'+str(variation_list[idx])

    start_at = vars()[obj_].n_testbefore - n_testbefore_list[variation_list[idx]]
    end_at = n_predict_list[variation_list[idx]] + vars()[obj_].n_testbefore

    start2 = vars()[obj_].n_train - vars()[obj_].n_testbefore
    end2 = vars()[obj_].n_train + vars()[obj_].n_predict

    x_axis = PLOT_SCALE*vars()[obj_].Delta_T_Sampling*np.arange(start_at - vars()[obj_].n_testbefore, 
                                                                end_at - vars()[obj_].n_testbefore ,
                                                                1)
    
    ax = vars()['ax_var'+str(variation_list[idx])+'_'+str(idx_ax2)]

    ax.plot(x_axis, vars()[obj_].predictions[start_at: end_at], 'o', c=us_colour_list[variation_list[idx]], alpha =0.5)#), markr_list[0], color = color_list[0], ms=10, alpha=0.5)
    ax.plot(x_axis, vars()[obj_].msmts[start2:end2][start_at: end_at], 'kx')#), markr_list[2], color = color_list[2], alpha=0.5, label=lbl_list[2])
    ax.plot(x_axis, vars()[obj_].truth[start2:end2][start_at: end_at], 'r')# color = color_list[2], alpha=0.5)
    
    ax.ticklabel_format(style='sci', scilimits=(0,2), axis='y')
    ax.axhline(0.0,  color='darkblue')#,label='Predict Zero Mean')
    ax.set(xlabel=r' Stps Fwd [num], $\Delta t =  %s$ [s]' %(vars()[obj_].Delta_T_Sampling), ylabel=r"Predictions [$f_n$]")
    ax.set_xlim([-50,100])
    ax.axvspan(-50,0, color='gray', alpha=0.3)
    ax.tick_params(direction='in', which='both')

########################################
# SPECIAL FIG: Single Prediction w Skip Msmts 
########################################

if DO_SKF == 'Yes':

    obj_0_ = 'obj_'+str(test_case_list[0])+'_'+str(variation_list[0])
    y_signal = vars()[obj_0_].msmts
    truth =  vars()[obj_0_].truth
    end_train = vars()[obj_0_].n_train

    for idx in xrange(NUM_SCENARIOS):

        obj_ = 'obj_'+str(test_case_list[idx])+'_'+str(variation_list[idx])
        skip = skip_list_2[variation_list[idx]]

        y_dummy_signal = np.zeros(vars()[obj_].number_of_points)
        y_dummy_signal[0:vars()[obj_].n_train] = y_signal[:end_train:skip]
        
        x_data_new, y_data_new, true_S_norm_new, predictions = vars()[obj_].return_SKF_skip_msmts(y_dummy_signal, 'newSKFfile'+str(skip), 'ZeroGain')

        start_at = vars()[obj_].n_testbefore - n_testbefore_list[variation_list[idx]]
        end_at = n_predict_list[variation_list[idx]] + vars()[obj_].n_testbefore
        start2 = vars()[obj_].n_train - vars()[obj_].n_testbefore
        end2 = vars()[obj_].n_train + vars()[obj_].n_predict

        x_axis = PLOT_SCALE*vars()[obj_].Delta_T_Sampling*np.arange(start_at - vars()[obj_].n_testbefore, 
                                                                    end_at - vars()[obj_].n_testbefore ,
                                                                    1)    
        ax_tui.plot(x_axis, predictions[start_at :end_at], 'o', c=us_colour_list[variation_list[idx]], alpha=0.5)
        if idx==0:
            ax_tui.plot(x_axis, truth[start2:end2], 'r')
            ax_tui.plot(x_axis, y_signal[start2:end2], 'kx')

        ax_tui.set(xlabel='Stps Fwd [num]', ylabel=r'$y_n [f_n] $')
        ax.set_xlim([-50, 100])
        ax_tui_labels.append('Some label')
        ax_tui.tick_params(direction='in', which='both')

        # Add Spectra

        ax_ = vars()['ax_var'+str(variation_list[idx])+'_'+str(2)]
        ax_.plot(x_data_new[0], y_data_new[0], '*', c=us_colour_list[variation_list[idx]])

        ax_.annotate('New Learned, Theory: %.2e, %.2e'%(np.round(np.sum(y_data_new[0])), true_S_norm_new), xy=(0.8,0.7), 
                    xycoords=('axes fraction', 'axes fraction'),
                    xytext=(1,1),
                    textcoords='offset points',
                    size=8,
                    color=us_colour_list[variation_list[idx]],
                    ha='right',
                    va='center')

    ax_tui.axvspan(-50,0, color='gray', alpha=0.3)
    ax_tui.axhline(1.0, linestyle='-', color='darkblue')

fig.savefig(savefigname+'.svg', format='svg')
plt.close(fig)