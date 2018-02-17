# The purpose of this script is to propagate information filtering equations. 

################################################################################
# Preamble
################################################################################

from __future__ import division, print_function, absolute_import

DATA_PATH = './' # for qif packpage
import sys 
sys.path.append(DATA_PATH)
from qif.common import calc_H


import numpy as np
import scipy.stats as stats
from scipy.special import erf as erf_func


################################################################################
# FISHER INFO RECURSIVE CALCULATIONS USING MID_TREAD / COIN FLIP SENSORS
################################################################################


def info_propagate(invP, V, S, J, invQ):

    '''Returns a single recusrive step of information filtering'''

    
    # print('invQ', invQ.shape, invQ)
    # print('V', V.shape, V)
    # print('S', S.shape, S)
    # print('invP', invP.shape, invP)
    # print('J', J.shape, J)
    
    invP_ = invQ + J - np.dot(np.dot(S.T, np.linalg.inv(invP + V)), S) # check transpose 

    return invP_
    


def info_execute(dynamical_model,x_signal, z_signal, sigma, R, P0, flag, process_noise_G=0):
    '''For time invariant dynamical, measurement and process noise matrix, we 
    propgate the infromation filtering equations'''
    
    
    order = dynamical_model.shape[0] # break for non-square dims
    stps = x_signal.shape[0] # break if this isn't z signal length
    
    if process_noise_G==0:
        process_noise_G = np.eye(order)
    
    # time invariant
    
    invQ = np.linalg.inv(np.dot(np.dot(process_noise_G.T, sigma*np.eye(order)), process_noise_G))
    V = np.dot(np.dot(dynamical_model.T, invQ), dynamical_model)
    S = -1*np.dot(dynamical_model.T, invQ)
    
    invP = np.zeros((stps, order, order))
    invP[0:order, :, :] = np.linalg.inv(np.eye(order)*P0) # initial condition for all time steps less than order. 
    
    # time varying
    
    for idx_stps in range(order, stps, 1): # start once enough data is seen
    
        x_apriori = x_signal[idx_stps - order: idx_stps ][::-1] # e.g. at idx_step =20, order =3, this will pick out x at steps = 19, 18, 17 such that x[0] is most recent
        # print('')
        # print('NEXT TIME STEP')
        # print("steps = ", idx_stps)
        # print('x_apriori', x_apriori)
        
        J = J_x_n(x_apriori, z_signal[idx_stps], R, flag)
        
        invP[idx_stps, :, :] = info_propagate(invP[idx_stps-1, :, :], V, S, J, invQ)
        
    return invP



def J_x_n(x_apriori, z_value, R, flag):
    ''' Returns the fisher information for the measurement model'''

    j_bit = J_bit(z_value, R, flag)    
    H_n = calc_H(x_apriori) # measurement model chosen in calc_H() function manually! this function outputs a Jacobian matrix.
    
    j_x_n =  j_bit*(np.outer(H_n, H_n)) #+ 0.00001*H_n[0]*np.eye(x_apriori.shape[0]))# nothing time invariant if H_n constant; dialing in an invertible Hn doesn't help
    
    # print('j_bit', j_bit.shape, j_bit)
    # print('H_n', H_n.shape, H_n)
    # print('j_x_n', j_x_n.shape, j_x_n)
    
    return j_x_n



def J_bit(z_value, R, flag):
    ''' Returns fisher information due to quantised measurements. The quantisation is 
    considered a coin flip, or a classical mid-riser quantisation'''

    if flag =='coin':
    
        j_bit = coinflip(z_value, R)
    
    else:
    
        j_bit = onebit(z_value, R)

    return j_bit



def coinflip(z_value, R, b=0.5):
    
    '''Returns J_{m=1, t} given by derived coin flip msmt action'''
    
    scale_R = np.real(np.sqrt(2*R))
    # rho_0 = erf_func(1.0/(scale_R)) + (scale_R/np.sqrt(np.pi))*(np.exp(-(1.0/scale_R)**2)- 1.0) # b= 1/2 - incorrect
    rho_0 = (2.0*b)*erf_func(2.0*b/(scale_R)) + (scale_R/np.sqrt(np.pi))*(np.exp(-(2.0*b/scale_R)**2)- 1.0) # b=1.0 - corrected feb 9 # recorrected b = 0.5 in feb 11
    
    if abs(z_value)== 0.5:
        # print("diverged, reset z = 0.49999999") # avoids divergence at the boundaries
        z_value=0.49999999 # INCORRECT - doesnt account for negative values - but doesnt matter for the next line
    
    j_onebit = (rho_0 * 4.0 ) / (1.0 - 4*(z_value)**2)
    return j_onebit



def onebit(z_value, R):
    
    ''' Returns probability of quantised outcome, y, given z,  i.e. p(y|z), where y = z + e.
    Measurement error e is zero mean, white and Gaussian distributed with variance rk (not standard deviation) . 
    In Karrlson (2005), p(y|z) \equiv Pr.(e < -z) \equiv Pr (e' < -z/ rk) for zero mean Gaussian normal e'''
    
    error_dist = stats.norm(loc=0, scale=1)
    normalised_value = -1*(z_value - 0.0) /np.sqrt(R) # normalised via standard deviation, take out 0.5 pi mean in x makes z zero mean
    rho_ = error_dist.cdf(normalised_value) # Theorum 3 
    j_onebit = np.exp(-(z_value**2)/R)*( 1./((rho_ * (1. - rho_)))) / (2.0 * np.pi*R)
    return j_onebit



###############################################################################
# CLRB CALCULATIONS FROM FISHER INFO
###############################################################################

def calculate_crlb(output_data, var, cflag_, name_R_, qubit_avg='median'):

    sv_data = output_data+'tc24_var_'+str(var)+'_flag_'+str(cflag_)+'_R_'+str(name_R_)+'.npz'
    data_object_1 = np.load(sv_data)
    
    raw_data_classical = np.mean(data_object_1['ensemble_crlb'], axis=0)
    
    if qubit_avg=='median':
        raw_data_coinflip = np.median(data_object_1['ensemble_crlb_coinflip'], 
                                     axis=0, overwrite_input=True)
    else:
        raw_data_coinflip = np.mean(data_object_1['ensemble_crlb_coinflip'], 
                                     axis=0, overwrite_input=True)
    
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
