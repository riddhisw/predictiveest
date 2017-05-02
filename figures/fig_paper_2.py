import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import numpy as np
import os

from analysis_tools.plot_KF import Plot_KF_Results
from analysis_tools.plot_BR import Plot_BR_Results
from analysis_tools.common import truncate_losses_
from analysis_tools.truth import Truth
from analysis_tools.kalman import Kalman

import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
# The purpose of this script is to produce a figure for 
# test_case_4, test_case_15 or test_case_16, while on the cluster.



test_case =7

savefig='Yes'
fsize=13.5
basis=0
bandwidth_=50
multiplier_= 20

us_colour_list = [0, 'dodgerblue', 'deepskyblue', 'b', 'darkblue', 'purple', 'maroon', 'deeppink', 'salmon','gold', 'olive', 'g', 'darkgreen', 'teal', 'lightslategreen']
skip = 1
max_forecast_loss=50
savetopath_ = '/scratch/RDS-FSC-QCL_KF-RW/Kalman/test_case_'+str(test_case)+'/'

max_stp_forwards_list=[]
max_stp_forwards_us_list=[]

# FIG:  Set template
gs = gridspec.GridSpec(2, 13, left=0.06, right=0.97, top=0.845, hspace=0.5, 
                       wspace=3.8, bottom=0.1)

fig_var = plt.figure(figsize=(18,6))
ax_main = fig_var.add_subplot(gs[0:,0:5])

subax = fig_var.add_axes([0.1, 0.475, 0.09, 0.275])

ax_loss1 = fig_var.add_subplot(gs[0, 5:7])
ax_loss2 = fig_var.add_subplot(gs[0, 7:9])
ax_loss3 = fig_var.add_subplot(gs[0, 9:11])
ax_loss4 = fig_var.add_subplot(gs[0, 11:13])

ax_kamp1 = fig_var.add_subplot(gs[1, 5:9])
ax_kamp2 = fig_var.add_subplot(gs[1, 9:13])


ax_loss_ = [ax_loss1, ax_loss2, ax_loss3, ax_loss4]
ax_kamp_ = [ax_kamp1, ax_kamp2]

idx_loss=0
idx_kamp=0
idx_pred=0

lbl_list = ['Prediction', 'Truth', 'Msmts']
markr_list = ['o', '-', 'x', ]
l_train = mpatches.Patch(color='gray', alpha=0.3)
ax_main_handlers = [l_train]
ax_main_label_list = [ 'Training']

ax_main.axvspan(-50,0, color='gray', alpha=0.3)
ax_main.axhline(1.0, linestyle='-', color='darkblue')
    
# Default Experimental Params
bandwidth_ = 50.0
FUDGE = 0.5
HILBERT_TRANSFORM = 2.0

variation_list = range(1,7,2)
for variation in variation_list:

    # Load data file paths
    filename0_ = 'test_case_'+str(test_case)+'_var_'+str(variation)
    filename_kf= filename0_+'_kfresults'
    filename_skippy = os.path.join(savetopath_, str(filename_kf)+'_skipmsmts_'+str(skip))
    filename_SKF = filename_skippy+str('SKF.npz')
    filename_truth = filename_kf+'_skip_msmts_Truth.npz'
    filename_BR = filename0_+str('BR_Map')
    filename_and_path_BR = os.path.join(savetopath_, str(filename_BR)+'.npz')


    inputs = np.load(os.path.join(savetopath_, filename_truth))
    kfdata = np.load(filename_SKF)
    br_obj = Plot_BR_Results(filename_and_path_BR)

    n_train = kfdata['n_train']
    n_predict =  kfdata['n_predict']
    n_testbefore = kfdata['n_testbefore']
    num_ = n_train + n_predict
    truth = inputs['truth']
    msmts = inputs['noisydata']
    instantA = kfdata['instantA']
    predictions = kfdata['predictions']
    exp_params_ = [n_train, n_predict, n_testbefore, multiplier_, bandwidth_]
    DeltaT = 1.0/(bandwidth_*multiplier_)

    true_noise_data = np.load(filename_and_path_BR)['true_noise_params']
    alpha = float(true_noise_data[1])
    f0 = float(true_noise_data[2])
    p = int(true_noise_data[3])
    J = int(true_noise_data[4])
    true_noise_params_ = [0, 'Uniform', alpha, f0, p, J, 1]

    bdelta = float(np.load(filename_and_path_BR)['kalman_params'][4])
    print('fo=', f0, 'bdelta=', bdelta)
    us = bdelta/f0
    max_stp_forwards_us_list.append(np.round(us, 2))
    
    # FIG X.0: Kalman Ensemble Average 
    kf_obj = Plot_KF_Results(exp_params_, filename_skippy+'.npz')
    kf_obj.load_data()

    max_stp_forwards_list.append(kf_obj.count_steps())

    start_at = n_testbefore - n_testbefore
    end_at = n_predict + n_testbefore
    x_axis = kf_obj.Delta_T_Sampling*np.arange(-n_testbefore, n_predict, 1)*1000

    ax_main.plot(x_axis, kf_obj.Normalised_Means_[0, start_at: end_at], '--', c=us_colour_list[variation])

    ax_main.set(xlabel='Stps Fwd [num]', ylabel=r'$log(E(err^2)$ [log(signal$^2$)]')
    ax_main.set_yscale('log')
    ax_main.set_ylim([10**(-5), 5])
    
    ax_main_label_list.append(r'KF Pred. ($\frac{\Delta \omega^B}{f_0}$ =  %s)'%(np.round(us, 2)))

    # FIG: Loss Variation       
    br_obj.load_data()
    br_obj.get_tuned_params(max_forecast_loss)
    
    
    for means_ind in xrange(2): # Creates two lists: list of index of (sigma, R), ordered by ascending loss values 
        vars()['x_br_params'+str(means_ind)], vars()['y_br_losses'+str(means_ind)] = truncate_losses_(br_obj.means_lists_[means_ind], br_obj.truncation)

    # Unzip (sigma, R) pairs
    sigma = [x[0] for x in br_obj.random_hyperparams_list]
    R = [x[1] for x in br_obj.random_hyperparams_list]

    ax = ax_loss_[idx_loss]
    ax.set_xscale('log')
    ax.set_yscale('log')
    for index in vars()['x_br_params'+str(0)]:
        ax.plot(sigma[index], R[index], 'o', c='tan', markersize=25, alpha=0.7)
    for index in vars()['x_br_params'+str(1)]:
        ax.plot(sigma[index], R[index], 'o', c='cyan', markersize=15, alpha=0.7)
    ax.plot(sigma, R, 'kv', markersize=5)#, label='Test Points')
    ax.plot(br_obj.lowest_pred_BR_pair[0], br_obj.lowest_pred_BR_pair[1], '*', color='crimson', markersize=15, mew=2)# label='Lowest Prediction Loss')
    ax.set_xlabel(r' Kalman $\sigma $ [signal$^2$]')
    ax.set_ylabel(r' Kalman $R$ [signal$^2$]')
    ax.set_xlim([10**-11,1000])
    ax.set_ylim([10**-11,1000])

    idx_loss +=1 


    # FIG X.3: True PSD vs. Kalman Amplitudes **2

    if variation==1 or variation==6 or variation==12:

        theory = Truth(true_noise_params_, num=num_, DeltaT=DeltaT)
        theory.beta_z_truePSD()
        x_data = [2.0*np.pi*kfdata['freq_basis_array'], theory.true_w_axis[theory.J -1:]]
        y_data = [(instantA**2)*(2*np.pi)*FUDGE, HILBERT_TRANSFORM*theory.true_S_twosided[theory.J -1:]]
        
        ax = ax_kamp_[idx_kamp]
        ax.set(xlabel=r'$\omega$ [rad]', ylabel=r'$S(\omega)$ [signal$^2$/(rad $s^{-1}$)]')
        ax.plot(x_data[0], y_data[0], 'o', c=us_colour_list[variation])#, label=' Pred.' (T. Pow: %s)'%(np.round(np.sum(y_data[0]))))
        ax.ticklabel_format(style='sci', scilimits=(0,2), axis='y')
        for label in ax.get_yticklabels():
            label.set_fontsize(fsize*0.8)
            label.set_color(us_colour_list[variation])

        ax.annotate('T.Pow: %s'%(np.round(np.sum(y_data[0]))), xy=(0.95,0.9), 
                    xycoords=('axes fraction', 'axes fraction'),
                    xytext=(1,1),
                    textcoords='offset points',
                    size=10,
                    color=us_colour_list[variation],
                    ha='right',
                    va='center')


        ax2 = ax.twinx()
        ax2.plot(x_data[1], y_data[1], 'r')#, label='Truth')#(T. Pow: %s)'%(np.round(theory.true_S_norm)))
        ax2.ticklabel_format(style='sci', scilimits=(0,2), axis='y')
        for label in ax2.get_yticklabels():
            label.set_fontsize(fsize*0.8)
            label.set_color('r')

        ax2.annotate('T.Pow: %s'%(np.round(theory.true_S_norm)), xy=(0.95,0.8), 
                    xycoords=('axes fraction', 'axes fraction'),
                    xytext=(1,1),
                    textcoords='offset points',
                    size=10,
                    color='r',
                    ha='right',
                    va='center')

        if idx_kamp==0:
            ax2.legend(bbox_to_anchor=(-0.1, 1.35, 1.0, 0.2), loc=2, ncol=1, frameon=False)

        idx_kamp +=1

# FIG: Inset
subax.set_ylim([1, 110])
subax.set_xscale('log')
subax.axhline(100.0,  color='brown', label='Max Pr.')
subax.set(xlabel=r'$\frac{\Delta \omega^B}{f_0}$' , ylabel="Parity [stps fwd]")
for idx_var in xrange(len(variation_list)):
    subax.plot(max_stp_forwards_us_list[idx_var], max_stp_forwards_list[idx_var], 'o', c=us_colour_list[variation_list[idx_var]])

for item in ([subax.title, subax.xaxis.label, subax.yaxis.label] + subax.get_xticklabels() + subax.get_yticklabels()):
    item.set_fontsize(12) 

for ax in [ax_main, ax_loss1, ax_loss2, ax_kamp1, ax_kamp2, ax_loss3, ax_loss4]:
    #ax.set(title="hello",xlabel="x", ylabel="y", ylim=[-1,1])
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fsize)

for label in ( ax_kamp1.get_yticklabels() + ax_kamp2.get_yticklabels() ):
    label.set_fontsize(fsize*0.8)

# LEGENDS

l_truth = mlines.Line2D([], [], linestyle='-', color='r')
l_msmt =  mlines.Line2D([], [], linestyle='None', color=None, marker='x', markerfacecolor='k', markeredgecolor='k', markersize=7)
l_pred0 = mlines.Line2D([], [], linestyle='-', color='darkblue')

for variation in variation_list:
    vars()['l_pred'+str(variation)] = mlines.Line2D([], [], linestyle='None', color=None, 
                                                    marker='o', markerfacecolor=us_colour_list[variation], 
                                                    markeredgecolor=us_colour_list[variation], markersize=7)
    ax_main_handlers.append(vars()['l_pred'+str(variation)])

ax_main_handlers.append(l_pred0)
ax_main_label_list.append('Predict Mean')
ax_main_handlers.append(l_truth)
ax_main_label_list.append('Truth')

ax_main.legend(handles=ax_main_handlers,
               labels=ax_main_label_list,
               bbox_to_anchor=(0.002, 1.02, 1.0, 0.2), loc=2, mode="expand",ncol=2, frameon=False)

test_circ = mlines.Line2D([], [], linestyle='None', color=None, marker='v', markerfacecolor='k', markeredgecolor='k', markersize=7)
pred_circ = mlines.Line2D([], [], linestyle='None',  color=None, marker='o', markerfacecolor='tan', markeredgecolor='tan', markersize=7)
fore_circ = mlines.Line2D([], [], linestyle='None',  color=None, marker='o', markerfacecolor='c', markeredgecolor='c', markersize=7)
pred_lowest = mlines.Line2D([], [], linestyle='None',  color=None, marker='*', markerfacecolor='crimson', markeredgecolor='crimson', markersize=7, alpha=1)
labels_traj = [r'Random Init. ($\sigma, R$)', 'Low Prediction Losses', 'Low Forecasting Losses', 'Min Prediction Loss', 'CA Optimum', 'COBLA Optimum']
ax_loss1.legend(handles=(test_circ, pred_circ, fore_circ, pred_lowest), 
                labels=labels_traj[0:4],  
                bbox_to_anchor=(-0.1, 1.35, 1.0, 0.2), loc=2, ncol=1, frameon=False)

if savefig=='Yes':
    fig_var.savefig(os.path.join(savetopath_, 'test_case_'+str(test_case))+'_paperfig3demo_.svg', format="svg")

plt.close(fig_var)

#    ax.plot(np.arange(-n_testbefore_, n_predict_, 1), kf_obj.Normalised_Means_[0, :], label=r'$\frac{\Delta s}{f_0}$ = %s' %(undersamp_strength[variation]))
