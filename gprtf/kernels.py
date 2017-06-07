'''
The purpose of this code is to provide a list of GPR Kernels. Kernels written in 
Tensor Flow will be used for training. Identical copies of Kernels in numpy
will be used for predictions. 

Kernel: "RQ"
            params[0] = beta -- variance multipler of stochastic process
            params[1] = l -- length scale
            params[2] = alpha -- scale mixture parameter
            
        Kernel: "RBF"
            params[0] = beta -- variance multipler of stochastic process
            params[1] = l -- length scale
            
        Kernel: "PER"
            params[0] = beta -- variance multipler of stochastic process
            params[1] = l -- length scale
            params[2] = p -- fundamental periodicity
            
        Kernel: "MAT" NOT DONE
            params[0] =  
            params[1] = 
            params[2] = 
    
        Measurment Noise Variance
            params[3] = msmt noise for Gaussian, uncorr RV [scalar,dtype=float64]

'''

import numpy as np
import tensorflow as tf # run script using pythonsys alias in command line


KER_NAME = {'RQ': rq ,'RBF': rbf, 'PER': periodic, 'MAT': matern}
KER_NUM_PARAMS = {'RQ': 3 ,'RBF': 2, 'PER': 3, 'MAT': 0} 


#### IN NUMPY ####

def kernel(kernel_name, hyp_params, R, x):
    '''
    Returns covariance matrix based on kernel choice,kernel hyper
    parameters, measurment noise and inputs:

    ARGS:
        kernel_name: One of [RQ, RBF, PER, MAT]
        hyp_params: Hyper parameters associated with kernel
        R: Measurement noise covariance strength
        x: Inputs, s.t. v == |x_i - x_j| for some i, j in length(x)
    
    Returns k: k[i,j] == R(v) + R*I, where I == identity
    '''
    length = x.shape[0]
    
    if hyp_params.shape[0] != KER_NUM_PARAMS[kernel_name]:
        print("Ill specified hyperparameters")
        raise RuntimeError

    k = KER_NAME[kernel_name](x, hyp_params) + R*np.diag(np.ones([length])
    return k

#### IN TENSORFLOW ####

