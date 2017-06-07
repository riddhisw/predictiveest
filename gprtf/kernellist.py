'''
This code lists kernels for GPR.
All code has two copies with identical functionality, written in:
    Tensor Flow - for training. 
    Numpy - for predictions. 
Functions for Tensor Flow are suffixed by _tf in the function name.

Kernel: "RQ"
            params[0] = sig_f -- variance multipler of stochastic process
            params[1] = l -- length scale
            params[2] = alpha -- scale mixture parameter
            
        Kernel: "RBF"
            params[0] = sig_f -- variance multipler of stochastic process
            params[1] = l -- length scale
            
        Kernel: "PER"
            params[0] = sig_f -- variance multipler of stochastic process
            params[1] = l -- length scale
            params[2] = p -- fundamental periodicity
            
        Kernel: "MAT" NOT DONE
            params[0] =  sig_f
            params[1] =  p -- order, s.t. we model AR(p) process
            params[2] = 
'''
# KER_NAME = {'RQ': rq ,'RBF': rbf, 'PER': periodic, 'MAT': matern}

import numpy as np
import tensorflow as tf 

######
# PER
######

def periodic(x, hyp_params):
    sig_f = hyp_params[0]
    l = hyp_params[1]
    p = hyp_params[2]
    v_matrix = x - x.T
    return sig_f * sig_f * np.exp(-2.0*np.power(np.sin(np.pi * v_matrix / p), 2) / (l*l))


######
# RBF
######

def rbf(x, hyp_params):
    sig_f = hyp_params[0]
    l = hyp_params[1]
    v_matrix = x - x.T
    return sig_f * sig_f * np.exp(-np.power(v_matrix, 2) / (2.0*l*l))


######
# MAT
######




######
# RQ
######

def rq(x, hyperparams):
    sig_f = hyp_params[0]
    l = hyp_params[1]
    alpha = hyp_params[2]
    v_matrix = x - x.T
    return sig_f * sig_f *(1 + (np.power(v_matrix, 2) / (2.0 * l * l * alpha)))**(-alpha)

