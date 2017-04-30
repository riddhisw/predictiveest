# The purpose of this script is to plot a single prediction, and learned 
# amplitudes, as well as those predicted by theory, for a single Kalman run.

import os
import numpy as np 
import matplotlib.pyplot as plt
from analysis_tools.truth import Truth
from analysis_tools.kalman import Kalman

#Filenames
test_case =7
savefig='Yes'
#variation=1
skip=1

for variation in [1, 2, 4, 6, 7]:

    #Experimental Params
    multiplier_list = [0, 20, 10, 6.66666666667, 5, 4, 2, 1.25]
    bandwidth_ = 50.0
        
    # Truth (to plot theoretical amplitudes)
    f0=10.0
    J_=4
    alpha=1.0
    apriori_f_mean = 0.0
    pdf_type = 'Uniform' 
    p = -1
    J = J_ + 1
    jstart = 1

    true_noise_params_ = [apriori_f_mean, pdf_type, alpha, f0, p, J_, jstart]

    # Load data
    savetopath_ =  '/scratch/RDS-FSC-QCL_KF-RW/Kalman/test_case_'+str(test_case)+'/' #'test_case_'+str(test_case)+'/' # '/scratch/RDS-FSC-QCL_KF-RW/Kalman/test_case_'+str(test_case)+'/'
    filename0_ = 'test_case_'+str(test_case)+'_var_'+str(variation)
    filename_kf= filename0_+'_kfresults'
    filename_skippy = os.path.join(savetopath_, str(filename_kf)+'_skipmsmts_'+str(skip))
    filename_SKF = filename_skippy+str('SKF.npz')
    filename_truth = filename_kf+'_skip_msmts_Truth.npz'

    inputs = np.load(os.path.join(savetopath_, filename_truth))
    data = np.load(filename_SKF)

    n_train = data['n_train']
    n_predict =  data['n_predict']
    n_testbefore = data['n_testbefore']
    num_ = n_train + n_predict

    truth = inputs['truth']
    msmts = inputs['noisydata']

    instantA = data['instantA']
    predictions = data['predictions']

    multiplier_= multiplier_list[variation]
    DeltaT = 1.0/(bandwidth_*multiplier_)

    # Generate true PSD
    theory = Truth(true_noise_params_, num=num_, DeltaT=DeltaT)
    theory.beta_z_truePSD()

    FUDGE = 0.5
    HILBERT_TRANSFORM = 2.0
    x_data = [2.0*np.pi*data['freq_basis_array'], theory.true_w_axis[theory.J -1:]]
    y_data = [(instantA**2)*(2*np.pi)*FUDGE, HILBERT_TRANSFORM*theory.true_S_twosided[theory.J -1:]]


    # Plot Results

    lbl_list = ['Prediction', 'Truth', 'Msmts']
    color_list = ['purple','red', 'black',]
    markr_list = ['o', '-', 'x', ]
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,10))

    ax = axes[0]
    Time_Axis = DeltaT*np.arange(-n_testbefore, n_predict, 1)
    predictions_list=[predictions, truth[n_train-n_testbefore:n_train+n_predict], msmts[n_train-n_testbefore:n_train+n_predict]]

    for i in xrange(3):
        ax.plot(Time_Axis, predictions_list[i], markr_list[i], color = color_list[i], alpha=0.5,label=lbl_list[i])
    ax.axhline(0.0,  color='black',label='Predict Zero')
    ax.axvline(0.0, linestyle='--', color='gray', label='Training Ends')

    ax.set(xlabel='Forecasting Steps > 0 ', ylabel="n-Step Ahead Msmt Prediction [Signal Units]")    
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", 
                borderaxespad=0, frameon=False, fontsize=14)

    ax = axes[1]
    ax.set(xlabel='Omega [radians]')
    ax.set_ylabel(r'$A_{KF}^2$ vs. PSD [Power/radians]')

    num_amps = len(x_data) -1
    for i in xrange(num_amps):
        ax.plot(x_data[i], y_data[i], markr_list[i], alpha=0.5, markersize=8.0,
                color=color_list[i], label=lbl_list[i]+', Power: %s'%(np.round(np.sum(y_data[i]))))
    ax.plot(x_data[num_amps], y_data[num_amps], 'r', label=lbl_list[1]+', Power: %s'%(np.round(theory.true_S_norm)))
            
    ax.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=1, mode="expand", 
                borderaxespad=0, frameon=False, fontsize=14)

    for ax in axes.flat:
        for item in (ax.get_xticklabels()):
            item.set_rotation('vertical')
        
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(14)

    fig.subplots_adjust(left=0.1, right=0.99, wspace=0.2, hspace=0.2, top=0.8, bottom=0.2)
    fig.suptitle('Theoretical Truth v. Learned Kalman Predictions Using Basis A', fontsize=14)


    if savefig=='Yes':
        fig.savefig(filename_skippy+'singlerun.svg', format="svg")        





