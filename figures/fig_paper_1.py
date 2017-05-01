# The purpose of this script is to generate summary figures

# Application on test_case_7, 8, 9, 10, 11, 12, 13, 14
import sys
sys.path.append('../')

import os
import numpy as np 
import matplotlib.pyplot as plt
from analysis_tools.plot_BR import Plot_BR_Results
from analysis_tools.common import truncate_losses_
from analysis_tools.truth import Truth
from analysis_tools.kalman import Kalman
from analysis_tools.plot_KF import Plot_KF_Results
import matplotlib.gridspec as gridspec

# Load test case and variation data

test_case =8
savefig='Yes'
p = -1
f0=10.0
J_=4
alpha=1.0
basis=0

skip = 1
max_forecast_loss=50
savetopath_ = '/scratch/RDS-FSC-QCL_KF-RW/Kalman/test_case_'+str(test_case)+'/'
multiplier_list = [0, 20, 10, 6.66666666667, 5, 4, 2, 1.25]

n_predict_list = [0, 100, 50, 33, 25, 20, 10, 7] # For plotting KF ensemble averages only
n_testbefore_list = [0, 50, 25, 17, 13, 10, 5, 3 ] # For plotting KF ensemble averages only
skip_list=[0, 1, 2, 3,  4, 5, 10, 16] # For plotting KF ensemble averages only



# FIG:  Set template
gs = gridspec.GridSpec(2,11, left=0.06, right=0.97, top=0.95, hspace=0.5, 
                       wspace=1.6, bottom=0.1)

fig_var = plt.figure(figsize=(18,6))
ax_main = fig_var.add_subplot(gs[0:, 0:5])

# subax = fig_var.add_axes([0.22, 0.19,0.2,0.4])

ax_loss1 = fig_var.add_subplot(gs[0, 5:7])
ax_kamp1 = fig_var.add_subplot(gs[0, 7:9])
ax_pred1 = fig_var.add_subplot(gs[0, 9:11])
ax_loss2 = fig_var.add_subplot(gs[1, 5:7])
ax_kamp2 = fig_var.add_subplot(gs[1, 7:9])
ax_pred2 = fig_var.add_subplot(gs[1, 9:11])

ax_main.axhline(1.0, linestyle='-', color='k', label='Predict Zero Mean')
ax_main.axvline(0.0, linestyle='--', color='gray', label='Training Ends')

ax_loss_ = [ax_loss1, ax_loss2]
ax_kamp_ = [ax_kamp1, ax_kamp2]
ax_pred_ = [ax_pred1, ax_pred2]

idx_loss=0
idx_kamp=0
idx_pred=0

lbl_list = ['Prediction', 'Truth', 'Msmts']
color_list = ['purple','red', 'black',]
markr_list = ['o', '-', 'x', ]
kf_colour_list = [0, 'b', 'g', 'orange', 'red', 'purple', 'cyan', 'gold']
    
# Default Experimental Params
bandwidth_ = 50.0
    
# Default Truth (to plot theoretical amplitudes)
apriori_f_mean = 0.0
pdf_type = 'Uniform' 
J = J_ + 1
jstart = 1
true_noise_params_ = [apriori_f_mean, pdf_type, alpha, f0, p, J_, jstart]
FUDGE = 0.5
HILBERT_TRANSFORM = 2.0

for variation in [1, 2, 4, 6, 7]:

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

    n_train = kfdata['n_train']
    n_predict =  kfdata['n_predict']
    n_testbefore = kfdata['n_testbefore']
    num_ = n_train + n_predict
    truth = inputs['truth']
    msmts = inputs['noisydata']
    instantA = kfdata['instantA']
    predictions = kfdata['predictions']
    multiplier_= multiplier_list[variation]
    exp_params_ = [n_train, n_predict, n_testbefore, multiplier_, bandwidth_]
    DeltaT = 1.0/(bandwidth_*multiplier_)

    # FIG X.0: Kalman Ensemble Average 
    kf_obj = Plot_KF_Results(exp_params_, filename_skippy+'.npz')
    kf_obj.load_data()
    start_at = n_testbefore - n_testbefore_list[variation]
    end_at = n_predict_list[variation] + n_testbefore
    x_axis = kf_obj.Delta_T_Sampling*np.arange(-n_testbefore_list[variation], n_predict_list[variation], 1)

    ax_main.plot(x_axis, kf_obj.Normalised_Means_[0, start_at: end_at], 'x--', c=kf_colour_list[variation], 
            label=r'$\Delta t$ =%s (Skipped Msmts = %s)'%(kf_obj.Delta_T_Sampling, skip_list[variation]))

    ax_main.set(xlabel='Time (s)', ylabel=r'Log(E[$e^2$]) [Log Signal Units^2]')
    ax_main.set_yscale('log')
    ax_main.set_ylim([10**(-5), 3])
    ax_main.legend(loc=0)


    # FIG X.2: One Loss Plot

    if variation==1 or variation==7:

        br_obj = Plot_BR_Results(filename_and_path_BR)
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
            ax.plot(sigma[index], R[index], 'ro', markersize=40, alpha=0.6)
        for index in vars()['x_br_params'+str(1)]:
            ax.plot(sigma[index], R[index], 'go', markersize=25, alpha=0.6)
        ax.plot(sigma, R, 'ko', markersize=10, label='Test Points')
        ax.plot(br_obj.lowest_pred_BR_pair[0], br_obj.lowest_pred_BR_pair[1], 'x', color='yellow',  markersize=25, mew=3, label='Lowest Prediction Loss')
        ax.set_xlabel(r' $\sigma $ [signal$^2$]')
        ax.set_ylabel(r' $R$ [signal$^2$]')
        ax.set_xlim([10**-11,1000])
        ax.set_ylim([10**-11,1000])

        idx_loss +=1 


    # FIG X.3: True PSD vs. Kalman Amplitudes **2

    if variation==1 or variation==7:

        theory = Truth(true_noise_params_, num=num_, DeltaT=DeltaT)
        theory.beta_z_truePSD()
        x_data = [2.0*np.pi*kfdata['freq_basis_array'], theory.true_w_axis[theory.J -1:]]
        y_data = [(instantA**2)*(2*np.pi)*FUDGE, HILBERT_TRANSFORM*theory.true_S_twosided[theory.J -1:]]
        ax = ax_kamp_[idx_kamp]
        ax.set(xlabel=r'$\omega$ [Rad]', ylabel=r'$S(\omega)$ [psd]')
        ax.plot(x_data[0], y_data[0], 'ko', label='Prediction, Power: %s'%(np.round(np.sum(y_data[0]))))
        ax.plot(x_data[1], y_data[1], 'r', label='Truth, Power: %s'%(np.round(theory.true_S_norm)))
        idx_kamp +=1

    # FIG X.4: Single Prediction
        
    if variation==1 or variation==7:

        ax = ax_pred_[idx_pred]
        Time_Axis = np.arange(-n_testbefore, n_predict, 1)
        predictions_list=[predictions, truth[n_train-n_testbefore:n_train+n_predict], msmts[n_train-n_testbefore:n_train+n_predict]]

        for i in xrange(3):
            ax.plot(Time_Axis, predictions_list[i], markr_list[i], color = color_list[i], alpha=0.5,label=lbl_list[i])
        ax.axhline(0.0,  color='black',label='Predict Zero')
        ax.axvline(0.0, linestyle='--', color='gray', label='Training Ends')

        ax.set(xlabel=r' Steps Fwd [num], $\Delta t = $ %s' %(DeltaT), ylabel="Signal [Signal Units]")    
        
        idx_pred+=1


for ax in [ax_main, ax_loss1, ax_loss2, ax_kamp1, ax_kamp2, ax_pred1, ax_pred2]:
    #ax.set(title="hello",xlabel="x", ylabel="y", ylim=[-1,1])
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(14)


if savefig=='Yes':
    fig_var.savefig(os.path.join(savetopath_, filename0_)+'_paperfig1_.svg', format="svg")
