import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import os
from analysis_tools.plot_KF import Plot_KF_Results
# The purpose of this script is to produce a figure for test_case_15 while on the cluster.

n_train_=2000
bandwidth_=50
n_predict_=100
n_testbefore_=50
multiplier_= 20

undersamp_strength=[800, 640, 500, 400, 320, 200, 160, 80, 20, 8, 2, 1, 0.5]

stp_fwd_limit=[]
test_case=15
skip=1
savetopath_ = '/scratch/RDS-FSC-QCL_KF-RW/Kalman/test_case_'+str(test_case)+'/'

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,8))
fsize=14
savefig='Yes'

ax.set_xlabel('Time Steps (s)')
ax.set_ylabel('Log(E[squared error]) (Log Signal Units^2 )')
ax.set_yscale('log')
ax.set_ylim([10**(-5), 10**3])

for variation in range(1, 14, 1):

    filename0_ = 'test_case_'+str(test_case)+'_var_'+str(variation)
    filename_kf= filename0_+'_kfresults'
    filename_skippy = os.path.join(savetopath_, str(filename_kf)+'_skipmsmts_'+str(skip))
    filename_SKF = filename_skippy+str('SKF.npz')
    filename_truth = filename_kf+'_skip_msmts_Truth.npz'
    filename_BR = filename0_+str('BR_Map')
    filename_and_path_BR = os.path.join(savetopath_, str(filename_BR)+'.npz')
    
    exp_params_ = [n_train_, n_predict_, n_testbefore_, multiplier_, bandwidth_]
    kf_obj = Plot_KF_Results(exp_params_, filename_skippy+'.npz')
    kf_obj.load_data()
    stp_fwd_limit.append(kf_obj.count_steps())

    ax.plot(kf_obj.Normalised_Means_[0, :], label='$\frac{Delta s}{f_0}$ = %s' %(undersamp_strength[variation]))

ax.legend(loc=2, fontsize=fsize)
ax.axhline(1.0, color = 'k', label='Predict Noise Mean')
ax.axvline(0.0, linestyle='--', color='gray', label='Training Ends')

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fsize)

if savefig=='Yes':
    fig.savefig(os.path.join(savetopath_, 'test_case_'+str(test_case))+'_paperfig2_.svg', format="svg")

plt.close(fig)
print('First figure done')

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,8))
ax.title('Max Steps Forwards vs. Undersampling Strength')
ax.plot(undersamp_strength[1:], stp_fwd_limit, 'ro')
ax.legend(loc=2, fontsize=fsize)
if savefig=='Yes':
    fig.savefig(os.path.join(savetopath_, 'test_case_'+str(test_case))+'_paperfig3_.svg', format="svg")

plt.close(fig)
print('2nd figure done')

