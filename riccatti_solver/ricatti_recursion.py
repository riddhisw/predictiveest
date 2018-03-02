################################################################################
# Preamble
################################################################################

from __future__ import division, print_function, absolute_import
import numpy as np
from riccatti_solver.qubitsensor import coinflip
from riccatti_solver.classicalsensor import onebit, onebit_trunc

DATA_PATH = './' # for qif packpage
import sys 
sys.path.append(DATA_PATH)
from qif.common import calc_H

################################################################################
# FISHER INFO RECURSIVE CALCULATIONS USING MID_TREAD / COIN FLIP SENSORS
################################################################################
def info_execute(dynamical_model,x_signal, z_signal, sigma, R, P0, flag, process_noise_G=0):
    '''For time invariant dynamical, measurement and process noise matrix, we 
    propagate the infromation filtering equations'''
    
    
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


def info_propagate(invP, V, S, J, invQ):

    '''Helper Function to info_execute
    Returns a single recursive step of information filtering'''
    
    invP_ = invQ + J - np.dot(np.dot(S.T, np.linalg.inv(invP + V)), S) # check transpose 

    return invP_


def J_x_n(x_apriori, z_value, R, flag):
    '''Helper Function to info_execute
    Returns the fisher information for the measurement model'''

    j_bit = J_bit(z_value, R, flag)    
    H_n = calc_H(x_apriori) # measurement model chosen in calc_H() function manually! this function outputs a Jacobian matrix.
    
    j_x_n =  j_bit*(np.outer(H_n, H_n)) #+ 0.00001*H_n[0]*np.eye(x_apriori.shape[0]))# nothing time invariant if H_n constant; dialing in an invertible Hn doesn't help
    
    # print('j_bit', j_bit.shape, j_bit)
    # print('H_n', H_n.shape, H_n)
    # print('j_x_n', j_x_n.shape, j_x_n)
    
    return j_x_n



def J_bit(z_value, R, flag):
    ''' Helper function to J_x_n: Returns fisher info due to sensor msmts.
    '''

    if flag =='coin':
    
        j_bit = coinflip(z_value, R)
    
    elif flag=='trunc':
    
        j_bit = onebit_trunc(z_value, R)

    else:
        
        j_bit = onebit(z_value, R)
    
    return j_bit
