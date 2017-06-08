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
            params[1] =  l
            params[2] = p -- order, s.t. we model AR(p) process
'''

import numpy as np
import tensorflow as tf 
PI = np.pi

def reshape_x(x):
    x = x.reshape([-1,1]) # -1 == unknown dim, 1 makes an internal new axis with only 1 element
    v_matrix = x - x.T
    return x, v_matrix

def reshape_x_tf(x):
    x = tf.reshape(x, [-1,1]) # -1 == unknown dim, 1 makes an internal new axis with only 1 element
    v_matrix = x - tf.transpose(x)
    return x, v_matrix

######
# PER
######

def periodic(x, hyp_params):
    x, v_matrix = reshape_x(x)   
    sig_f = hyp_params[0]
    l = hyp_params[1]
    p = hyp_params[2]
    return sig_f * sig_f * np.exp(-2.0*np.power(np.sin(np.pi * v_matrix / p), 2) / (l*l))

def periodic_tf(x, hyp_params):
    x, v_matrix = reshape_x_tf(x)
    sig_f = hyp_params[0]
    l = hyp_params[1]
    p = hyp_params[2]
    return sig_f * sig_f * tf.exp(-2.0*tf.pow(tf.sin(PI * v_matrix / p), 2) / (l*l))   

######
# RBF
######

def rbf(x, hyp_params):
    x, v_matrix = reshape_x(x)  
    sig_f = hyp_params[0]
    l = hyp_params[1]
    return sig_f * sig_f * np.exp(-np.power(v_matrix, 2) / (2.0*l*l))

def rbf_tf(x, hyp_params):
    x, v_matrix = reshape_x_tf(x)  
    sig_f = hyp_params[0]
    l = hyp_params[1]
    return sig_f * sig_f * tf.exp(-tf.pow(v_matrix, 2) / (2.0*l*l))
######
# RQ
######

def rq(x, hyperparams):
    x, v_matrix = reshape_x(x)  
    sig_f = hyp_params[0]
    l = hyp_params[1]
    alpha = hyp_params[2]

    return sig_f * sig_f *(1 + (np.power(v_matrix, 2) / (2.0 * l * l * alpha)))**(-alpha)

def rq_tf(x, hyperparams):
    x, v_matrix = reshape_x_tf(x)  
    sig_f = hyp_params[0]
    l = hyp_params[1]
    alpha = hyp_params[2]

    return sig_f * sig_f *tf.pow((1 + (tf.pow(v_matrix, 2) / (2.0 * l * l * alpha))), -1.0*alpha)
######
# SM
######


######
# MAT
######
 # not doing this since the order (p) is too large for efficient computation

 # too many hyper parameters to optimize