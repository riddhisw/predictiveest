################################################################################
# Preamble
################################################################################

from __future__ import division, print_function, absolute_import
import numpy as np



###############################################################################
# CLRB CALCULATIONS FROM FISHER INFO
###############################################################################

def calculate_crlb(output_data, var, cflag_, name_R_, qubit_avg='median'):

    sv_data = output_data+'tc24_var_'+str(var)+'_flag_'+str(cflag_)+'_R_'+str(name_R_)+'.npz'
    data_object_1 = np.load(sv_data)
    
    raw_data_classical = np.mean(data_object_1['ensemble_crlb'], axis=0)
    
    if qubit_avg=='median':
        raw_data_coinflip = np.median(data_object_1['ensemble_crlb_coinflip'], 
                                     axis=0)
    else:
        raw_data_coinflip = np.mean(data_object_1['ensemble_crlb_coinflip'],
                                     axis=0)
    
    crlb_classical = calc_variance(raw_data_classical)
    crlb_coinflip = calc_variance(raw_data_coinflip)
    
    return crlb_classical, crlb_coinflip


def calc_variance(fisher_stream, axis=0):
    
    shp = fisher_stream.shape
    variance_stream = np.zeros(shp)
    
    for idx_stp in xrange(shp[axis]):
    
        # reset dynamic slicing + pick out the idx_stp matrix along axis
        slc = [slice(None)]*len(shp)
        slc[axis] = slice(idx_stp, idx_stp+1, 1)
        
        variance_stream[slc] = np.linalg.inv(fisher_stream[slc])

    return variance_stream
