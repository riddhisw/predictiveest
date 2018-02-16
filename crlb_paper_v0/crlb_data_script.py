import sys
import os

test_case = int(sys.argv[1])
var_R_ = float(sys.argv[2])
name_R_ = str(sys.argv[3])
cflag_ = int(sys.argv[4])

add_crlb_noise = 0.0
if cflag_ == 1:
    add_crlb_noise = var_R_*1.0
    

import numpy as np

###############################################################################
# FILE PATH AND IMPORT INFO
###############################################################################

sys.path.append('/project/RDS-FSC-QCL_KF-RW/crlb/')
input_datapath='/scratch/RDS-FSC-QCL_KF-RW/CRLB/input_data/'
output_data = '/scratch/RDS-FSC-QCL_KF-RW/CRLB/output_data/' 

from qif.common import generate_AR, noisy_z
from akf.armakf import get_autoreg_model
from data_tools.load_raw_cluster_data import LoadExperiment
from riccatti_solver.ricatti_recursion import info_execute

ver=0

###############################################################################
# DATA PARAMETERS
###############################################################################

variation_scan=[1, 2, 3, 4, 7]

# Number of trials in an ensemble 
runs = 50

# Process noise 
true_oe = 0.001

# Initial variance for Kalman filter
p0init = 1000000

# Number of AR values to disregard in the beginning
burn_in = 500

# Total length of measurement record
number_of_steps = 2000

###############################################################################
# RUN SCRIPT
###############################################################################

for idx_var in xrange(len(variation_scan)):

    experiment = 0
    experiment = LoadExperiment(test_case, variation_scan[idx_var], 
                                QKF_load='No',
                                QKF_path='',
                                LKFFB_load ='No',
                                LKFFB_path='',
                                LSF_load='No',
                                AKF_load='Yes', 
                                AKF_path=input_datapath,
                                GPRP_load='No')

    order = experiment.AKF_weights.shape[0]
    
    dynamical_model = get_autoreg_model(order, experiment.AKF_weights) 

    ensemble_crlb = np.zeros((runs, number_of_steps, 101, 101))
    ensemble_crlb_coinflip = np.zeros((runs, number_of_steps, 101, 101))
    
    sv_data = output_data+'tc24_var_'+str(variation_scan[idx_var])+'_flag_'+str(cflag_)+'_R_'+str(name_R_)+'.npz'

    for idx_runs in xrange(runs):

        x_init = np.random.uniform(low=-0.5*np.pi, high=0.5*np.pi, size=order)
        
        true_x = generate_AR(x_init, number_of_steps + burn_in, 
                             experiment.AKF_weights, true_oe)[burn_in:] 
        noisy_z_ = noisy_z(true_x, add_crlb_noise) 

        ensemble_crlb[idx_runs, :, :, :] = info_execute(dynamical_model,
                                              true_x, noisy_z_, 
                                              true_oe, var_R_, p0init, 'classical')

        ensemble_crlb_coinflip[idx_runs, :, :, :] = info_execute(dynamical_model,
                                              true_x, noisy_z_, 
                                              true_oe, var_R_, p0init, 'coin')


        if idx_runs%3 == 0 or idx_runs == runs-1:
            np.savez(sv_data, 
                     ensemble_crlb=ensemble_crlb,
                     ensemble_crlb_coinflip=ensemble_crlb_coinflip)

