################################################################################
# Preamble
################################################################################

from __future__ import division, print_function, absolute_import

import numpy as np
import scipy.stats as stats
from scipy.special import erf 

################################################################################
# Mid Tread Quantiser
################################################################################

def onebit(z_value, R):
    
    ''' Returns probability of quantised outcome, y, given z,  i.e. p(y|z), where y = z + e.
    Measurement error e is zero mean, white and Gaussian distributed with variance rk (not standard deviation) . 
    In Karrlson (2005), p(y|z) \equiv Pr.(e < -z) \equiv Pr (e' < -z/ rk) for zero mean Gaussian normal e'''
    
    error_dist = stats.norm(loc=0, scale=1)
    normalised_value = -1*(z_value - 0.0) /np.sqrt(R) # normalised via standard deviation, take out 0.5 pi mean in x makes z zero mean
    rho_ = error_dist.cdf(normalised_value) # Theorum 3 
    j_onebit = np.exp(-(z_value**2)/R)*( 1./((rho_ * (1. - rho_)))) / (2.0 * np.pi*R)
    return j_onebit


################################################################################
# Mid Tread Quantiser with a Truncated Error Model
################################################################################

def Xi(z, R, b):
    
    """Returns normalised pdf for mean zero and variance R Gaussian errors but defined over a finite interval paramterised by b.
    Eqn (C9) in Fullnotes v3."""
    
    result = (1./(4.*b))*erf((b-z)/np.sqrt(2.*R)) + (1./(4.*b))*erf((b+z)/np.sqrt(2.*R)) 

    return result


def pheadsT(z, R, b):
    
    """Returns unnormalised probability of being in a top level for a single bit midtread quantiser with truncated errors"""
    
    erf_terms = (b+z)*erf((b+z)/np.sqrt(2.*R)) - (b-z)*erf((b-z)/np.sqrt(2.*R)) + (2.*b)*erf((2.*b)/np.sqrt(2*R))
    exp_terms = np.exp(-(((b+z)/np.sqrt(2*R)))**2) - np.exp(-(((b-z)/np.sqrt(2*R)))**2) + np.exp(-((2.*b/np.sqrt(2*R)))**2) - 1.0
    
    result = (1./(4*b))*erf_terms + (1./(4*b))*np.sqrt(2.*R/np.pi)*exp_terms
    
    return result


def ptailsT(z, R, b):
    
    """Returns unnormalised probability of being in a bottom level for a single bit midtread quantiser with truncated errors"""
    
    return pheadsT(-z, R, b)


def onebit_trunc(z, R, b=0.5):
    
    """Fisher Information for a single time step for a midtread single bit amplitude quantiser with truncated
    Gaussian error model """
    
    result = (Xi(-z, R, b))**2/(pheadsT(z, R, b)*ptailsT(z, R, b))
    
    return result