import sys
import os

test_case = int(sys.argv[1])
var_R_ = float(sys.argv[2])
name_R_ = str(sys.argv[3])
cflag_ = int(sys.argv[4])


import numpy as np

###############################################################################
# FILE PATH AND IMPORT INFO
###############################################################################

sys.path.append('/project/RDS-FSC-QCL_KF-RW/crlb/')
output_data = '/scratch/RDS-FSC-QCL_KF-RW/CRLB/output_data/' 
from riccatti_solver.ricatti_recursion import calculate_crlb

###############################################################################
# DATA PARAMETERS
###############################################################################

variation_scan=[1, 2, 3, 4, 7]
N=2000
order=101

###############################################################################
# RUN SCRIPT
###############################################################################

varlen=len(variation_scan)
midtrd = np.zeros((varlen, N, order, order))
cnflp = np.zeros_like(midtrd)

for idx_var in xrange(varlen):

    try:
        midtrd[idx_var, :, :,:], cnflp[idx_var,:, :,:] = calculate_crlb(output_data, 
                                                                        variation_scan[idx_var], 
                                                                        cflag_, name_R_)
        
        sv_data = output_data+'CRLB_tc24_flag_'+str(cflag_)+'_R_'+str(name_R_)+'.npz'
        np.savez(sv_data, midtrd=midtrd, cnflp=cnflp)
    
    except:
        print(name_R_, idx_var, 'FAILED')

