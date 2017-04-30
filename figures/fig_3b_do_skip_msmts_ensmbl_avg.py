import os
import numpy as np
import matplotlib.pyplot as plt
from analysis_tools.plot_KF import Plot_KF_Results

# This script can be used for Test_Case_7, Test_Case_8, and Test_Case_9
# 
# 
# This script produces ensemble average results for Kalman Filtering vs. predict zero
# We plot data from two sources:
# (A) Skip msmts implemented via setting W==0 during recursion
# (B) Skip msmts implemented via changing experimental sampling rates

test_case=8
skip_list=[1, 2, 4, 5, 10, 16]
var_list=[1, 2, 4,  5, 6, 7]


# Load experiment
n_train_ = [0, 2000, 1000, 667, 500, 400, 200, 125]
n_predict_ = 100
n_predict_list = [0, 100, 50, 33, 25, 20, 10, 7]
n_testbefore_list = [0, 50, 25, 17, 13, 10, 5, 3 ]
n_testbefore_ = 50
multiplier_ = [0, 20, 10, 6.66666666667, 5, 4, 2, 1.25]
bandwidth_ = 50.0

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,8))

for idx_skip in xrange(6):

        skip= skip_list[idx_skip]
        variation_num=var_list[idx_skip] 

        # Load data for (A)
        # Variation == 1 
        savetopath_ = '/scratch/RDS-FSC-QCL_KF-RW/Kalman/test_case_'+str(test_case)+'/'
        filename0_ = 'test_case_'+str(test_case)+'_var_'+str(1)
        filename_skippy = os.path.join(savetopath_, filename0_+'_kfresults_skipmsmts_'+str(skip))

        exp_params_ = [n_train_[1], n_predict_, n_testbefore_, multiplier_[1], bandwidth_]
        skip_via_zero_gain = Plot_KF_Results(exp_params_, filename_skippy+'.npz')
        skip_via_zero_gain.load_data()

        # Load data for (B)
        # For test_cases = 7, 8, 9, the variation num selects parameters according to 
        # KalmanParameterRegimes.ods
        # skip_msmt = [1,2,3,4,5,10,16] == variation_num=[1, 2, 3, 4,5,6,7]

        filename0_ = 'test_case_'+str(test_case)+'_var_'+str(variation_num)
        filename_skippy = os.path.join(savetopath_, filename0_+'_kfresults_skipmsmts_'+str(1))

        exp_params_ = [n_train_[variation_num], n_predict_, n_testbefore_, multiplier_[variation_num], bandwidth_]
        skip_via_fs = Plot_KF_Results(exp_params_, filename_skippy+'.npz')
        skip_via_fs.load_data()

        # Show one prediction
        
        skip_via_fs.show_one_prediction()
        skip_via_zero_gain.show_one_prediction()
        
        # Plot 

        figname=str(os.path.join(savetopath_, filename0_))

        choice_counter=0
        kf_colour_list = ['b', 'g', 'orange', 'red', 'purple', 'cyan', 'gold']
        kf_choice_labels =['Skip msmts via Sampling Rates', 'Skip msmts via Zero Gain']
        step_forward_zerogain_limit = skip_via_zero_gain.count_steps()
        step_forward_fs_limit = skip_via_fs.count_steps()
        fsize=14
        savefig='Yes'
        print(step_forward_fs_limit, step_forward_zerogain_limit)

        ax.set_title(figname)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Log(E[squared error]) (Log Signal Units^2 )')
        ax.set_yscale('log')
        ax.set_ylim([10**(-5), 10**3])

        # Plot of Sampling Rate Data
        start_at = n_testbefore_ - n_testbefore_list[variation_num]
        end_at = n_predict_list[variation_num] + n_testbefore_

        x_axis = skip_via_fs.Delta_T_Sampling*np.arange(-n_testbefore_list[variation_num], n_predict_list[variation_num], 1)

        ax.plot(x_axis, skip_via_fs.Normalised_Means_[choice_counter, start_at: end_at], 'x', c=kf_colour_list[idx_skip],
        label=r'$\Delta t$ =%s, Skip Msmts = %s'%(skip_via_fs.Delta_T_Sampling, skip_list[idx_skip])) #kf_choice_labels[0]+' (parity @ '+str(step_forward_fs_limit)+' stps fwd)')

        ## Plot of Data via Zero Gain 

        x_axis = skip_via_zero_gain.Delta_T_Sampling*np.arange(-n_testbefore_, n_predict_, 1)
        ax.plot(x_axis, skip_via_zero_gain.Normalised_Means_[choice_counter, :], '--', c=kf_colour_list[idx_skip],
                label='via Zero Gain')#kf_choice_labels[1]+' (parity @ '+str(step_forward_zerogain_limit)+' stps fwd)')

ax.legend(loc=2, fontsize=fsize)
ax.plot(x_axis, skip_via_zero_gain.Normalised_Predict_Zero_Means_, c = 'k', label='Predict Noise Mean')
ax.axvline(0.0, linestyle='--', color='gray', label='Training Ends')

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fsize)
#if savefig=='Yes':
    #fig.savefig(filename0_+'_compare_skipmsmt_.svg', format="svg")

plt.show()


