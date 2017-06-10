import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from plotting_tools.case_data_explorer import CaseExplorer as cs


########################################
# INPUT DATA 
########################################

path_to_directory = './'#'/scratch/RDS-FSC-QCL_KF-RW/Kalman'
savefigname = 'tc_7_'
ADD_LS_DATA = 'Yes'
DO_SKF = 'Yes'

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

gs = gridspec.GridSpec(4, NUM_SCENARIOS + 2,
                       left=0.05, right=0.97, 
                       top=0.9, bottom=0.05, 
                       wspace=0.75, hspace=0.65)

fig = plt.figure(figsize=( 3.0*(NUM_SCENARIOS + 2), 3.0*4))

count=0

ax_kea = fig.add_subplot(gs[0:2,0:2])

if DO_SKF == 'Yes':
    ax_tui = fig.add_subplot(gs[2:4,0:2])

for idx in xrange((NUM_SCENARIOS)):
    for idx_ax2 in xrange(4):
        if variation_list[idx]==7 and DO_SKF=='Yes':
            vars()['ax_var'+str(variation_list[idx])+'_'+str(idx_ax2)] = fig.add_subplot(gs[idx_ax2, 2+idx], facecolor='mistyrose')
        else:
            vars()['ax_var'+str(variation_list[idx])+'_'+str(idx_ax2)] = fig.add_subplot(gs[idx_ax2, 2+idx])
        #vars()['ax_var'+str(variation_list[idx])+'_'+str(idx_ax2)].locator_params(axis='x', numticks=4)

########################################
# FIG: Custom Legends
########################################
optimal_star = 'magenta'

l_train = mpatches.Patch(color='gray', alpha=0.3)
predictzeroline =  mlines.Line2D([], [], linestyle='-', color='darkblue')

randinit = mlines.Line2D([], [], linestyle='None', color=None, marker='v', markerfacecolor='k', markeredgecolor='k', markersize=7)
optimalstar = mlines.Line2D([], [], linestyle='None',  color=None, marker='*', markerfacecolor=optimal_star, markeredgecolor=optimal_star, markersize=7, alpha=1)

pred_circ = mlines.Line2D([], [], linestyle='None',  color=None, marker='o', markerfacecolor='tan', markeredgecolor='tan', markersize=7)
pred_line = mlines.Line2D([], [], linestyle='-',  color=optimal_star)

fore_circ = mlines.Line2D([], [], linestyle='None',  color=None, marker='o', markerfacecolor='c', markeredgecolor='c', markersize=7)
fore_line = mlines.Line2D([], [], linestyle='-',  color='teal')

truthline = mlines.Line2D([], [], linestyle='-', color='r')
noisy_x = mlines.Line2D([], [], linestyle='None', color=None, marker='x', markerfacecolor='k', markeredgecolor='k', markersize=7)


vars()['ax_var'+str(variation_list[2])+'_'+str(0)].legend(handles=(randinit,  optimalstar, pred_circ, fore_circ),
                                                                  labels= [r'Random Init. ($\sigma, R$)', r'Tuned ($\sigma, R$)', 'Low State Est. Risk', 'Low Prediction Risk'],
                                                                  bbox_to_anchor=(-3.8, 1.16, 4.0, 0.2), loc=2, ncol=4, frameon=True, fontsize=13.5,
                                                                  facecolor='linen',
                                                                  edgecolor='white')

vars()['ax_var'+str(variation_list[2])+'_'+str(1)].legend(handles=(pred_line, fore_line, pred_circ, fore_circ), 
                                                                  labels=['Tuned KF - State Est. Risk', 'Tuned KF - Pred. Risk', r'Rand. ($\sigma, R$) - State Est. Risk', r' Rand. ($\sigma, R$) - Pred. Risk'],  
                                                                  bbox_to_anchor=(-3.8, 1.16, 4.0, 0.2), loc=2, ncol=4, frameon=True, 
                                                                  facecolor='linen',
                                                                  edgecolor='white',
                                                                  fontsize=13.5)

vars()['ax_var'+str(variation_list[2])+'_'+str(2)].legend(handles=(truthline, noisy_x), 
                                                                  labels=['Truth', 'Noisy Data'],  
                                                                  bbox_to_anchor=(-3.8, 1.16, 4.0, 0.2), loc=2, ncol=2, frameon=True, fontsize=13.5,
                                                                  facecolor='linen',
                                                                  edgecolor='white')

########################################
# FIG: Size, Colors and Labels
########################################
fsize=13.5
PLOT_SCALE = 1000
savefig='Yes'
us_colour_list = ['g', 'dodgerblue', 'purple', 'maroon', 'darkorange']

loss_color_list = ['tan', 'c', optimal_star, 'teal']
style = ['o--', 'o--', '-', '-']
ax_kea_labels=['A', 'B', 'C', 'D', 'E']
ax_tui_labels=['A*', 'B*', 'C*', 'D*', 'E*']

########################################
# FIG: Plot Kalman Averages on First Plot
########################################

for idx in xrange(NUM_SCENARIOS):

    obj_ = 'obj_'+str(test_case_list[idx])+'_'+str(variation_list[idx])
    max_stp_fwd.append(vars()[obj_].count_steps())

    start_at = vars()[obj_].n_testbefore - n_testbefore_list[variation_list[idx]]
    end_at = n_predict_list[variation_list[idx]] + vars()[obj_].n_testbefore

    x_axis = PLOT_SCALE*vars()[obj_].Delta_T_Sampling*np.arange(start_at - vars()[obj_].n_testbefore, 
                                                                end_at - vars()[obj_].n_testbefore ,
                                                                1)

    ax_kea.plot(x_axis, vars()[obj_].Normalised_Means[BASIS_PRED_NUM,start_at :end_at],
                '--',
                label=ax_kea_labels[idx],
                c=us_colour_list[idx])

    if DO_SKF=='Yes' and ADD_LS_DATA == 'Yes' and idx==0:
        ls_data = '/ls_data_for_plotting/ls_norm_tc_'+str(test_case_list[idx])+'_var_'+str(variation_list[idx])+'.npz'
        norm_means = np.load(path_to_directory+ls_data)['Normalised_Means']
        ax_kea.plot(x_axis[n_testbefore_list[variation_list[idx]]: n_testbefore_list[variation_list[idx]] + 50], norm_means,'k', label='A (LS)') # max LS n_predict =50
    
    elif DO_SKF !='Yes' and ADD_LS_DATA == 'Yes':
        ls_data = '/ls_data_for_plotting/ls_norm_tc_'+str(test_case_list[idx])+'_var_'+str(variation_list[idx])+'.npz'
        norm_means = np.load(path_to_directory+ls_data)['Normalised_Means']
        ax_kea.plot(x_axis[n_testbefore_list[variation_list[idx]]: n_testbefore_list[variation_list[idx]] + 50], norm_means,
                '-',
                label=ax_kea_labels[idx]+' (LS)',
                c=us_colour_list[idx])
    
    ax_kea.set_yscale('log')
    ax_kea.set_ylim([10**(-5), kea_max])
    
    ax_kea.tick_params(direction='in', which='both')

    ax_kea.set_xlim([-60, stps_fwd_truncate_])
    xtickslabels =[x.get_text() for x in ax_kea.get_xticklabels()]
    xtickslabels[0] = str(r'$-n_{T}$')
    print(xtickslabels)
    xtickvalues = [int(x) for x in ax_kea.get_xticks()]
    xtickslabels[1:] = xtickvalues[1:]
    print(xtickslabels)
    ax_kea.set_xticklabels(xtickslabels)
    ax_kea.set(xlabel='Stps Fwd [num]', ylabel=r' Norm. log($\langle (f_n -\hat{f_n})^2 \rangle_D$) [log($f_n^2$)]')

ax_kea.axvspan(-60,-50, color='gray', alpha=0.6)
ax_kea.axvspan(-50,0, color='gray', alpha=0.3, label="Training")
ax_kea.axhline(1.0, linestyle='-', color='darkblue', label='Predict Mean')
ax_kea.legend(bbox_to_anchor=(0.002, 1.05, 1.0, 0.2), loc=2, mode="expand",ncol=3, frameon=False, fontsize=fsize)   


######################################
# Add broken lines to x_axis
######################################

from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                  AnnotationBbox)

arr_img = plt.imread('broken_axis_2.png', format='png')

imagebox = OffsetImage(arr_img, zoom=0.18)
imagebox.image.axes = ax_kea
xy = (-49.5, 0.00001)

ab = AnnotationBbox(imagebox, xy,
                    xybox=(0.,0.),
                    frameon=False,
                    xycoords='data',
                    boxcoords="offset points",
                    #pad=0.5,
                    )

ax_kea.add_artist(ab)

########################################
# FIG: Loss Map on Variational Plots
########################################

idx_ax2 = 0# Loss Map

for idx in xrange(NUM_SCENARIOS):

    obj_ = 'obj_'+str(test_case_list[idx])+'_'+str(variation_list[idx])
    kf, akf = vars()[obj_].return_low_loss_hyperparams_list()
    s0, R0, s1, R1, sigma, R, p_index, p_losses = [item for item in kf]
    
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

    ax_.annotate(ax_kea_labels[idx], xy=(0, 1.45), 
            xycoords=('axes fraction', 'axes fraction'),
            xytext=(1,1),
            textcoords='offset points',
            size=24,
            color=us_colour_list[idx],
            ha='left',
            va='center')


########################################
# FIG: Loss Histograms on Variational Plots
########################################

idx_ax2 = 1 # Loss Histogram

for idx in xrange(NUM_SCENARIOS):

    obj_ = 'obj_'+str(test_case_list[idx])+'_'+str(variation_list[idx])
    kf, akf = vars()[obj_].return_low_loss_hyperparams_list()
    s0, R0, s1, R1, sigma, R, p_index, p_losses = [item for item in kf]

    ax_ = vars()['ax_var'+str(variation_list[idx])+'_'+str(idx_ax2)]

    start_at = vars()[obj_].n_testbefore - n_testbefore_list[variation_list[idx]] #== 0 if Delta_T is not changing
    end_at = n_predict_list[variation_list[idx]] #==  n_predict if Delta_T is not changing
    x_axis = PLOT_SCALE*vars()[obj_].Delta_T_Sampling*np.arange(start_at - vars()[obj_].n_testbefore, end_at, 1)
    opt_index = p_index[0]
    unopt_index = list(set(range(0, vars()[obj_].num_randparams, 1)) - set(p_index))[0] #just choosing the first one

    count=0
    for chosen_index in [unopt_index, opt_index]:
        ax_.plot(x_axis[:n_testbefore_list[variation_list[idx]]], np.mean(vars()[obj_].macro_prediction_errors[chosen_index,:,start_at:], axis=0), 
                style[count], markersize=4, c=loss_color_list[count])
        count+=1
        ax_.plot(x_axis[n_testbefore_list[variation_list[idx]] :], np.mean(vars()[obj_].macro_forecastng_errors[chosen_index,:,0:end_at], axis=0), 
                style[count], markersize=4, c=loss_color_list[count])
        count+=1
    
    ax_.set_yscale('log')
    ax_.set_xlim([-50,50])
    ax_.set_ylim([loss_hist_min, loss_hist_max])
    ax_.axvspan(-50,0, color='gray', alpha=0.3)
    
    if idx==0:
        ax_.set_ylabel(r'log($\langle (f_n -\hat{f_n})^2 \rangle_D$) [log($f_n^2$)]')
    if idx==2:
        ax_.set_xlabel('Stps Fwd [num]')
    
    ax_.tick_params(direction='in', which='both')
    

########################################
# FIG: Amplitude Graphs for Variation Plots
########################################

idx_ax2 = 3 # Amplitudes from Single Run

for idx in xrange(NUM_SCENARIOS):

    obj_ = 'obj_'+str(test_case_list[idx])+'_'+str(variation_list[idx])
    x_data, y_data, true_S_norm = vars()[obj_].return_fourier_amps()

    ax = vars()['ax_var'+str(variation_list[idx])+'_'+str(idx_ax2)]

    # This factor scales Kalman Amplitudes when undersampling is present.
    # The scaling is necessary else the true PSD is much smaller than Kalman Amplitudes
    # The undersampled regimes forces the filter to weight Kalman frequencies much higher 
    # to compensate for the true frequencies it can't see.
    
    UNDSAMPL_FUDGE = 0.0
    UNDSAMPL_FUDGE = vars()[obj_].kalman_params[4]/vars()[obj_].f0
    if UNDSAMPL_FUDGE < 1.0:
        UNDSAMPL_FUDGE=1.0 # This correction doesn't apply to adequately sampled regimes
    print("UNDSAMPL_FUDGE=", UNDSAMPL_FUDGE)

    ax.plot(x_data[0], y_data[0]*(1.0/UNDSAMPL_FUDGE), 'o', c=us_colour_list[idx])
    
    #ax.ticklabel_format(style='sci', scilimits=(0,2), axis='y')
    ax.set_yscale('log')
    ax.set_ylim([amp_PSD_min, 1])
    ax.tick_params(direction='in', which='both')

    ax.annotate('T.Pow: %.2e'%(np.round(np.sum(y_data[0]))), xy=(0.8, 1.11), 
                xycoords=('axes fraction', 'axes fraction'),
                xytext=(1,1),
                textcoords='offset points',
                size=10,
                color=us_colour_list[idx],
                ha='right',
                va='center')

    ax.plot(x_data[1], y_data[1], 'r')

    bandedge = vars()[obj_].f0*(vars()[obj_].J-1)*2.0*np.pi
    compedge = vars()[obj_].bandwidth*2.0*np.pi
    ax.axvline(x=bandedge, ls='--', c='r', label= 'True Band Edge')
    ax.axvline(x=compedge, ls='--', c='k', label= 'KF Basis Ends')

    if idx==0:
        ax.set_ylabel(r'$S(\omega)$ [$f_n^2$/(rad $s^{-1}$)]')
        ax.legend(bbox_to_anchor=(-0.3, 1.3, 4.0, 0.2), loc=2, ncol=2, frameon=True, fontsize=13.5,
                                                                  facecolor='linen',
                                                                  edgecolor='white')
    if idx==2:
        ax.set_xlabel(r'$\omega$ [rad]')

########################################
# FIG: Single Prediction in Natural Timescale 
########################################
idx_ax2 = 2 # Single Prediction in Natural Timescale


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

    ax.plot(x_axis, vars()[obj_].predictions[start_at: end_at], 'o', c=us_colour_list[idx], alpha =1.0)#), markr_list[0], color = color_list[0], ms=10, alpha=0.5)
    ax.plot(x_axis, vars()[obj_].msmts[start2:end2][start_at: end_at], 'kx')#), markr_list[2], color = color_list[2], alpha=0.5, label=lbl_list[2])
    ax.plot(x_axis, vars()[obj_].truth[start2:end2][start_at: end_at], 'r')# color = color_list[2], alpha=0.5)
    
    ax.ticklabel_format(style='sci', scilimits=(0,2), axis='y')
    ax.axhline(0.0,  color='darkblue')#,label='Predict Zero Mean')
    
    ax.set_xlim([-50,stps_fwd_truncate_])
    ax.axvspan(-50,0, color='gray', alpha=0.3)
    ax.tick_params(direction='in', which='both')

    if idx==0:
        ax.set_ylabel(r'Predictions [$f_n$]')
    if idx==2:
        ax.set_xlabel('Stps Fwd [num]')
    

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
        ax_tui.plot(x_axis, predictions[start_at :end_at], '*', 
                    label=ax_tui_labels[idx],
                    c=us_colour_list[idx], 
                    alpha=0.5)
        if idx==0:
            ax_tui.plot(x_axis, truth[start2:end2], 'r')
            ax_tui.plot(x_axis, y_signal[start2:end2], 'kx')

        ax_tui.set(xlabel='Stps Fwd [num]', ylabel=r'$y_n [f_n] $')
        ax.set_xlim([-50, 100])
        
        ax_tui.tick_params(direction='in', which='both')

        ax_tui.set_xlim([-60, stps_fwd_truncate_])
        xtickslabels =[x.get_text() for x in ax_kea.get_xticklabels()]
        xtickslabels[0] = str(r'$-n_{T}$')
        print(xtickslabels)
        xtickvalues = [int(x) for x in ax_tui.get_xticks()]
        xtickslabels[1:] = xtickvalues[1:]
        print(xtickslabels)
        ax_tui.set_xticklabels(xtickslabels)

        # Add Spectra

        ax_ = vars()['ax_var'+str(variation_list[idx])+'_'+str(3)]
        
        UNDSAMPL_FUDGE = 0.0
        UNDSAMPL_FUDGE = vars()[obj_].kalman_params[4]/vars()[obj_].f0
        if UNDSAMPL_FUDGE < 1.0:
            UNDSAMPL_FUDGE=1.0 # This correction doesn't apply to adequately sampled regimes
        print("UNDSAMPL_FUDGE=", UNDSAMPL_FUDGE)

        ax_.annotate('T.Pow: %.2e (*) '%(np.round(np.sum(y_data_new[0]))), 
                    xy=(0.96, 1.03), 
                    xycoords=('axes fraction', 'axes fraction'),
                    xytext=(1,1),
                    textcoords='offset points',
                    size=10,
                    color=us_colour_list[idx],
                    ha='right',
                    va='center')
    
    ax_tui.axvspan(-50,0, color='gray', alpha=0.3)
    ax_tui.axvspan(-60,-50, color='gray', alpha=0.6)
    ax_tui.axhline(1.0, linestyle='-', color='darkblue')
    ax_tui.legend(bbox_to_anchor=(0.002, 0.9, 1.0, 0.2), loc=2, mode="expand",ncol=5, frameon=False, fontsize=fsize)    

    ######################################
    # Add broken lines to x_axis
    ######################################
    imagebox2 = OffsetImage(arr_img, zoom=0.18)
    imagebox2.image.axes = ax_tui
    xy = (0.093, 0)

    ab2 = AnnotationBbox(imagebox2, xy,
                        xybox=(0.,0.),
                        frameon=False,
                        xycoords='axes fraction',
                        boxcoords="offset points",
                        #pad=0.5,
                        )

    ax_tui.add_artist(ab2)



######################################
# Font Sizes
######################################

if DO_SKF == 'Yes':
    main_ax_list = [ax_kea, ax_tui]
elif DO_SKF != 'Yes':
    main_ax_list = [ax_kea]

for ax in main_ax_list:
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fsize)

for idx in xrange(NUM_SCENARIOS):
    for idx_ax2 in xrange(4):
        ax = vars()['ax_var'+str(variation_list[idx])+'_'+str(idx_ax2)]
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fsize)
#        for item in ([ax.xaxis.label, ax.yaxis.label]):
#            item.set_weight('bold')


######################################
# Little Kia (Inset)
######################################   
# FIG: Inset

kia = fig.add_axes([0.08, 0.72, 0.08, 0.14], facecolor='white')

#kia.set_xlim([0.1, 10**3])
#kia.set_xscale('log')

kia.set_ylim([1, n_predict_list[1]+10])
#kia.set_yscale('log')

#kia.axvspan(1, 10**3,  color='linen', label=r'$f_n$ undersampled')
kia.axvspan(1, 2,  color='mistyrose',label='r < 2')
kia.plot(tc7, [100, 50, 25, 10, 7] , '--', c='brown', label='Equal $t$ ')
kia.legend(loc=4)


kia.axhline(n_predict_list[1],  color='brown')
kia.set(xlabel=dial_label , ylabel="Parity [stps fwd]")
kia.xaxis.tick_top()
kia.xaxis.set_label_position('top')
kia.tick_params(direction='in', which='both')

idx_var=0
for idx_var in xrange(NUM_SCENARIOS):
    kia.plot(dial[idx_var], max_stp_fwd[idx_var], 'o', c=us_colour_list[idx_var], markersize=10)

for item2 in ([kia.title, kia.xaxis.label, kia.yaxis.label] + kia.get_xticklabels() + kia.get_yticklabels()):
    item2.set_fontsize(11) 

######################################
# Save and Close
######################################   
#    
fig.savefig(savefigname+'.png', format='png')
plt.close(fig)
