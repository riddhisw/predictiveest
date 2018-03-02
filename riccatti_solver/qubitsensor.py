################################################################################
# Preamble
################################################################################

from __future__ import division, print_function, absolute_import

import numpy as np
from scipy.special import erf # as erf_func


def coinflip(z_value, R, b=0.5):
    
    '''Returns J_{m=1, t} given by derived coin flip msmt action'''
    
    scale_R = np.real(np.sqrt(2*R))
    # rho_0 = erf_func(1.0/(scale_R)) + (scale_R/np.sqrt(np.pi))*(np.exp(-(1.0/scale_R)**2)- 1.0) # b= 1/2 - incorrect
    rho_0 = rho_qubit(R,b)
    
    if abs(z_value)== 0.5:
        # print("diverged, reset z = 0.49999999") # avoids divergence at the boundaries
        z_value=0.49999999 # INCORRECT - doesnt account for negative values - but doesnt matter for the next line
    
    j_onebit = (rho_0 * 4.0 ) / (1.0 - 4*(z_value)**2)
    return j_onebit


def rho_qubit(R, b=0.5): # b=1.0 - corrected feb 9 # recorrected b = 0.5 in feb 11
    
    "Scaling factor for the Fisher Information of a qubit sensor"
    
    scale_R = np.real(np.sqrt(2*R))
    result = (2.0*b)*erf(2.0*b/(scale_R)) + (scale_R/np.sqrt(np.pi))*(np.exp(-(2.0*b/scale_R)**2)- 1.0) 
    
    return result