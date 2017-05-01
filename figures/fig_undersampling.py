import sys
sys.path.append('../')

import matplotlib.pyplot as plt
import os
from analysis_tools.plot_KF import Plot_KF_Results
# The purpose of this script is to produce a figure for test_case_5 while on the cluster.

n_train_=2000
n_predict_=50
bandwidth_=50
n_predict_=100
n_testbefore_=50

multiplier_=[0, 20.0, 18.0, 16.0, 14.0, 12.0, 10.0, 5.0, 4.0, 3.0, 2.0, 1.8, 1.25, 1.0]
undersamp_strength=[0, 563, 506, 450, 394, 337, 281, 141, 113, 84, 56, 51, 35, 28]

stp_fwd_limit=[]
test_case=5
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
    filename_skippy = os.path.join(savetopath_, filename0_+'_kfresults_skipmsmts_'+str(skip))
    
    exp_params_ = [n_train_, n_predict_, n_testbefore_, multiplier_[variation], bandwidth_]
    run = Plot_KF_Results(exp_params_, filename_skippy+'.npz')

    run.load_data()
    stp_fwd_limit.append(run.count_steps())

    ax.plot(run.Normalised_Means_[0, :], label='$\frac{Delta s}{f_0}$ = %s' %(undersamp_strength[variation]))

ax.legend(loc=2, fontsize=fsize)
ax.axhline(1.0, color = 'k', label='Predict Noise Mean')
ax.axvline(0.0, linestyle='--', color='gray', label='Training Ends')

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(fsize)

if savefig=='Yes':
    fig.savefig(savetopath_+'hello_i_am_here.svg', format="svg")

print('First figure done')

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,8))
ax.title('Max Steps Forwards vs. Undersampling Strength')
ax.plot(undersamp_strength[1:], stp_fwd_limit, 'ro')
ax.legend(loc=2, fontsize=fsize)
if savefig=='Yes':
    fig.savefig(savetopath_+'hello_i_am_still_here.svg', format="svg")

print('2nd figure done')

