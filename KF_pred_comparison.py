import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from analysis_tools.case_data_explorer import CaseExplorer as cs
from kf.armakf import autokf as akf 
from kf.fast_2 import kf_2017 as kf 
from analysis_tools.testcaseDict import tcDict

path_to_directory = sys.argv[1]
dict_key = sys.argv[2]
savefigname = dict_key

test_case_list = tcDict[dict_key][0]
variation_list = tcDict[dict_key][1]
dial = tcDict[dict_key][2]
dial_label = tcDict[dict_key][3]
n_predict_list = tcDict[dict_key][4]
n_testbefore_list = tcDict[dict_key][5]


########################################
# REFERENCE PARAMETERS (NO CHANGE) 
########################################
ADD_LS_DATA='Yes'
DO_SKF='No'
max_stp_fwd=[]

loss_hist_min = 10**-2
loss_hist_max = 10**6
amp_PSD_min = 10**-10
stps_fwd_truncate_=50
kea_max = 10**3

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

gs = gridspec.GridSpec(3, NUM_SCENARIOS ,
                       left=0.05, right=0.97, 
                       top=0.9, bottom=0.05, 
                       wspace=0.3, hspace=0.45)

fig = plt.figure(figsize=( 5.0*(NUM_SCENARIOS), 3.0*4))

count=0
ROWS =3
for idx in xrange((NUM_SCENARIOS)):
    for idx_ax2 in xrange(ROWS):
        if variation_list[idx]==7 and DO_SKF=='Yes':
            vars()['ax_var'+str(variation_list[idx])+'_'+str(idx_ax2)] = fig.add_subplot(gs[idx_ax2, idx], facecolor='mistyrose')
        else:
            vars()['ax_var'+str(variation_list[idx])+'_'+str(idx_ax2)] = fig.add_subplot(gs[idx_ax2, idx])
        #vars()['ax_var'+str(variation_list[idx])+'_'+str(idx_ax2)].locator_params(axis='x', numticks=4)

########################################
# FIG: Size, Colors and Labels
########################################
fsize=13.5
PLOT_SCALE = 1000
savefig='Yes'
us_colour_list = ['g', 'dodgerblue', 'purple', 'maroon', 'darkorange']

style = ['-', '-', '-', '-']
ax_kea_labels=['A', 'B', 'C', 'D', 'E']
ax_tui_labels=['A*', 'B*', 'C*', 'D*', 'E*']

########################################
# FIG: Single Predictions Comparison
########################################

idx_ax0 = 0 # Time predictions on the first row
idx_ax1 = 1 # Amp predictions on the second row
idx_ax2 = 2 # Ensemble Avg predictions on the third row

for idx in xrange(NUM_SCENARIOS):

    ax = vars()['ax_var'+str(variation_list[idx])+'_'+str(idx_ax0)]
    obj_ = 'obj_'+str(test_case_list[idx])+'_'+str(variation_list[idx])
    
    # Get data
    output = vars()[obj_].return_low_loss_hyperparams_list(truncation_=2) # this has data from kf and akf
    y_signal = vars()[obj_].msmts
    truth =  vars()[obj_].truth
    end_train = vars()[obj_].n_train

    kf_omega, kf_amp, kf_Snorm, kf_pred =  vars()[obj_].return_SKF_skip_msmts(y_signal, 'newSKFfile'+str(1), 'ZeroGain')
    akf_x, akf_y, akf_y_norm, akf_pred  = vars()[obj_].return_AKF(y_signal)
    ls_pred = vars()[obj_].return_LS(y_signal)
    
    start_at = vars()[obj_].n_testbefore - n_testbefore_list[variation_list[idx]]
    end_at = n_predict_list[variation_list[idx]] + vars()[obj_].n_testbefore
    start2 = vars()[obj_].n_train - vars()[obj_].n_testbefore
    end2 = vars()[obj_].n_train + vars()[obj_].n_predict

    x_axis = PLOT_SCALE*vars()[obj_].Delta_T_Sampling*np.arange(start_at - vars()[obj_].n_testbefore, 
                                                                    end_at - vars()[obj_].n_testbefore ,
                                                                    1)    
    ax.plot(x_axis, kf_pred[start_at :end_at], 'o', 
                label='KF - Basis of Oscillators',
                c=us_colour_list[idx],
                markersize = 5,
                alpha=0.8)

    ax.plot(x_axis, akf_pred[start_at :end_at], 'o', 
            label='KF - Autoregressive AR(q=101) with LS Weights',
            c='k', 
            markersize = 5,
            alpha=0.8)

    fudge = n_predict_list[variation_list[idx]]

    if fudge > 50:
        fudge = 50

    ax.plot(x_axis[n_testbefore_list[variation_list[idx]]: n_testbefore_list[variation_list[idx]] + fudge], ls_pred[:n_predict_list[variation_list[idx]]], 
            '-', 
            label='LS',
            c=us_colour_list[idx],
            markersize = 5,
            alpha=1.0)

    ax.plot(x_axis, vars()[obj_].msmts[start2:end2][start_at: end_at], 'kx', label='Msmts')#), markr_list[2], color = color_list[2], alpha=0.5, label=lbl_list[2])
    ax.plot(x_axis, vars()[obj_].truth[start2:end2][start_at: end_at], 'r', label = 'Truth')# color = color_list[2], alpha=0.5)
    
    ax.ticklabel_format(style='sci', scilimits=(0,2), axis='y')
    ax.axhline(0.0,  color='darkblue')#,label='Predict Zero Mean')
    
    ax.set_xlim([-50,stps_fwd_truncate_])
    ax.axvspan(-50,0, color='gray', alpha=0.3, label="Training")
    ax.tick_params(direction='in', which='both')
    #ax.legend(loc=2)

    ax.annotate(ax_kea_labels[idx], xy=(0, 1.3), 
            xycoords=('axes fraction', 'axes fraction'),
            xytext=(1,1),
            textcoords='offset points',
            size=24,
            color=us_colour_list[idx],
            ha='left',
            va='center')
    
    if idx==0:
        ax.set_ylabel(r'Predictions [$f_n$]')
        ax.legend(bbox_to_anchor=(-0.3, 1.06, 4.0, 0.2), loc=2, ncol=6, frameon=True, fontsize=13.5,
                   facecolor='linen',
                   edgecolor='white')
    if idx==2:
        ax.set_xlabel('Stps Fwd [num]')
    

    ########################################
    # FIG: Amplitude Graphs 
    ########################################

    ax1 = vars()['ax_var'+str(variation_list[idx])+'_'+str(idx_ax1)]

    # This factor scales Kalman Amplitudes when undersampling is present.
    # The scaling is necessary else the true PSD is much smaller than Kalman Amplitudes
    # The undersampled regimes forces the filter to weight Kalman frequencies much higher 
    # to compensate for the true frequencies it can't see.
    
    UNDSAMPL_FUDGE = 0.0
    UNDSAMPL_FUDGE = vars()[obj_].kalman_params[4]/vars()[obj_].f0
    if UNDSAMPL_FUDGE < 1.0:
        UNDSAMPL_FUDGE=1.0 # This correction doesn't apply to adequately sampled regimes
    #print("UNDSAMPL_FUDGE=", UNDSAMPL_FUDGE)

    ax1.plot(kf_omega[0], kf_amp[0]*(1.0/UNDSAMPL_FUDGE), 'o', c=us_colour_list[idx])

    ax1.set_yscale('log')
    ax1.set_ylim([amp_PSD_min, 100]) # Starts log scale at amp_PSD_min
    ax1.tick_params(direction='in', which='both')

    print('KF AMP ', type(kf_amp), kf_amp[0].shape)

    ax1.annotate('T.Pow KF: %.3e'%(np.sum(kf_amp[0])), xy=(0.95, 1.1), 
                xycoords=('axes fraction', 'axes fraction'),
                xytext=(1,1),
                textcoords='offset points',
                size=10,
                color=us_colour_list[idx],
                ha='right',
                va='center')

    ax1.plot(kf_omega[1], kf_amp[1], 'r') # Truth

    print('AKF AMP ', type(akf_y_norm), akf_y_norm.shape)

    akf_cut_off_idx = int(float(akf_x.shape[0])/(akf_x[-1]/300.0)) # S(w) from AR(q) weights trucnated at omega = 300 rad
    #ax1.plot(akf_x[0:akf_cut_off_idx], akf_y[0:akf_cut_off_idx], 'kx') # Unnormalised S(w) from weights, 
    if akf_cut_off_idx > 0 :
        ax1.plot(akf_x[0:akf_cut_off_idx], akf_y_norm[0:akf_cut_off_idx], 'ko', markersize=5) 
    elif akf_cut_off_idx == 0:
        ax1.plot(akf_x, akf_y_norm, 'ko', markersize=5) 
    ax1.annotate('T.Pow AKF: %.3e'%(np.sum(akf_y_norm)), xy=(0.95, 1.05), 
                xycoords=('axes fraction', 'axes fraction'),
                xytext=(1,1),
                textcoords='offset points',
                size=10,
                color='k',
                ha='right',
                va='center')

    bandedge = vars()[obj_].f0*(vars()[obj_].J-1)*2.0*np.pi
    compedge = vars()[obj_].bandwidth*2.0*np.pi
    ax1.axvline(x=bandedge, ls='--', c='r', label= 'True Band Edge')
    ax1.axvline(x=compedge, ls='--', c='k', label= 'KF Basis Ends')

    if idx==0:
        ax1.set_ylabel(r'$S(\omega)$ [$f_n^2$/(rad $s^{-1}$)]')
        ax1.legend(bbox_to_anchor=(-0.3, 1.11, 4.0, 0.2), loc=2, ncol=2, frameon=True, fontsize=13.5,
                                                                facecolor='linen',
                                                                edgecolor='white')
    if idx==2:
        ax1.set_xlabel(r'$\omega$ [rad]')


    ########################################
    # FIG: Ensemble Avg Graphs
    ########################################

    ax2 = vars()['ax_var'+str(variation_list[idx])+'_'+str(idx_ax2)]

    predict_zero_err = vars()[obj_].macro_truth # err^2 for zero mean processes
    predict_zero_ensmbl = np.mean(predict_zero_err**2 , axis=1) # mean over ensemble runs
    normalised_means = np.zeros(vars()[obj_].n_testbefore + vars()[obj_].n_predict)

    for idx_kf_type in xrange(2):

        s0, R0, s1, R1, sigma, R, p_index, p_losses = [item for item in output[idx_kf_type]]

        # Timescales
        start_at = vars()[obj_].n_testbefore - n_testbefore_list[variation_list[idx]]
        end_at = n_predict_list[variation_list[idx]] + vars()[obj_].n_testbefore

        x_axis = PLOT_SCALE*vars()[obj_].Delta_T_Sampling*np.arange(start_at - vars()[obj_].n_testbefore, 
                                                        end_at - vars()[obj_].n_testbefore ,
                                                        1)

                
        # choose predict zero mean for optimal index
        opt_index = p_index[0]
        predict_zero_opt_run = predict_zero_ensmbl[opt_index, vars()[obj_].n_train - vars()[obj_].n_testbefore : ]

        # Choose dataset (KF v. AKF):
        PRED_DICT = {'0': np.mean(vars()[obj_].macro_prediction_errors, axis=1),
                     '1': np.mean(vars()[obj_].akf_macro_prediction_errors, axis=1)}
        FORE_DICT = {'0': np.mean(vars()[obj_].macro_forecastng_errors, axis=1), 
                     '1': np.mean(vars()[obj_].akf_macro_forecastng_errors, axis=1)}
        
        #get bayes loss for optimal index for one type of Kalman filter
        bayesloss_p = PRED_DICT[str(idx_kf_type)][opt_index,:]
        bayesloss_f = FORE_DICT[str(idx_kf_type)][opt_index,:]

        # normalise prediction means
        normalised_means[0: vars()[obj_].n_testbefore] = bayesloss_p/predict_zero_opt_run[0: vars()[obj_].n_testbefore]
        normalised_means[vars()[obj_].n_testbefore:] = bayesloss_f/predict_zero_opt_run[vars()[obj_].n_testbefore:]
        test_line = predict_zero_opt_run/predict_zero_opt_run

        print("idx =", idx, idx_kf_type, normalised_means.shape)

        colorchoice = us_colour_list[idx]
        if idx_kf_type != 0:
            colorchoice = 'k'

        ax2.plot(x_axis, normalised_means[start_at : end_at], 'o',
                alpha=0.8,
                markersize=5,
                c=colorchoice)

        if idx_kf_type ==0:
            ax2.plot(x_axis, test_line[start_at : end_at], '*--',
                    c='darkblue',
                    markersize=5,
                    alpha=0.8,
                    label='Predict Mean (Zero)')    


    ls_data = '/ls_data_for_plotting/ls_norm_tc_'+str(test_case_list[idx])+'_var_'+str(variation_list[idx])+'.npz'
    ls_norm_means = np.load(path_to_directory+ls_data)['Normalised_Means']

    predict_list = n_predict_list[variation_list[idx]]
    if n_predict_list[variation_list[idx]] > 50:
        predict_list = 50    
    ax2.plot(x_axis[n_testbefore_list[variation_list[idx]]: n_testbefore_list[variation_list[idx]] +  predict_list ], ls_norm_means[0: predict_list ],
            '-', 
            alpha=0.8,
            markersize=5,
            c=us_colour_list[idx]) # max LS n_predict =50
    
    ax2.set_yscale('log')
    ax2.set_ylim([10**(-5), kea_max])
    ax2.tick_params(direction='in', which='both')
    ax2.set_xlim([-60, stps_fwd_truncate_])
    xtickslabels =[x.get_text() for x in ax2.get_xticklabels()]
    xtickslabels[0] = str(r'$-n_{T}$')
    print(xtickslabels)
    xtickvalues = [int(x) for x in ax2.get_xticks()]
    xtickslabels[1:] = xtickvalues[1:]
    print(xtickslabels)
    ax2.set_xticklabels(xtickslabels)
    ax2.axvspan(-75,-50, color='gray', alpha=0.6)
    ax2.axvspan(-50, 0, color='gray', alpha=0.3)

    if idx==2:
        ax2.set(xlabel='Stps Fwd [num]')

    if idx==0:
        ax2.set(ylabel=r' Norm. log($\langle (f_n -\hat{f_n})^2 \rangle_D$) [log($f_n^2$)]')
        ax2.legend(bbox_to_anchor=(-0.3, 1.06, 4.0, 0.2), loc=2, ncol=6, frameon=True, fontsize=13.5,
                                                        facecolor='linen',
                                                        edgecolor='white')
    
    #ax2.axhline(1.0, linestyle='-', color='darkblue', label='Predict Mean')

    #if idx==0:
    #    ax2.legend(bbox_to_anchor=(0.002, 1.1, 4.0, 0.2), loc=2, mode="expand",ncol=7, frameon=False, fontsize=fsize)   
    
    ######################################
    # Add broken lines to x_axis
    ######################################

    from matplotlib.offsetbox import (TextArea, DrawingArea, OffsetImage,
                                    AnnotationBbox)

    arr_img = plt.imread('broken_axis_2.png', format='png')

    imagebox = OffsetImage(arr_img, zoom=0.18)
    imagebox.image.axes = ax2
    xy = (-49.5, 0.00001)

    ab = AnnotationBbox(imagebox, xy,
                        xybox=(0.,0.),
                        frameon=False,
                        xycoords='data',
                        boxcoords="offset points",
                        #pad=0.5,
                        )

    ax2.add_artist(ab)

for idx in xrange(NUM_SCENARIOS):
    for idx_ax in xrange(ROWS):
        ax = vars()['ax_var'+str(variation_list[idx])+'_'+str(idx_ax)]
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fsize)

fig.savefig(savefigname+'pred_comparison.png', format='png')
plt.close(fig)