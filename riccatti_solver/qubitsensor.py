################################################################################
# Preamble
################################################################################

from __future__ import division, print_function, absolute_import

import numpy as np
from scipy.special import erf 


def coinflip(z_value, R, b=0.5):
    
    '''Returns J_{vartheta=1, n} given by derived coin flip msmt action'''

    rho_0 = rho_qubit(R,b)
    
    if abs(z_value)== 0.5:
        z_value=0.49999999 # -ve values irrelevant as z^2 term req in Fisher info
    
    j_onebit = (rho_0 * 4.0 ) / (1.0 - 4*(z_value)**2)
    return j_onebit


def rho_qubit(R, b):
    
    "Scaling factor, rho_0, for the Fisher Information of a qubit sensor"
    
    scale_R = np.real(np.sqrt(2*R))
    result = erf(2.0*b/(scale_R)) + (scale_R/(2.0*b*np.sqrt(np.pi)))*(np.exp(-(2.0*b/scale_R)**2)- 1.0) 
    
    return result