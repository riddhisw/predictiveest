'''
This code lists kernels for GPR.
All code has two copies with identical functionality, written in:
    Tensor Flow - for training. 
    Numpy - for predictions. 
Functions for Tensor Flow are suffixed by _tf in the function name.

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
'''

import numpy as np
import tensorflow as tf 

