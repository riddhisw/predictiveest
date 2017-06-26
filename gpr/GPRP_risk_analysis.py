"""
Created on Thu Apr 20 19:20:43 2017

@author: riddhisw

PACKAGE: gpr
MODULE: gpr.GPRP_tuned_runs.py

The purpose of gpr is to implement Gaussian Process Regression (GPR) using GPy.
At present, the kernel being testing is the Periodic Kernel. L-BFGS-B is used
to optimise predictions in each run. 

MODULE PURPOSE: Returns predictions from GPy for Periodic Kernel; with L-BFGS-B tuning

CLASSES: GPRP_Optimisation
METHODS: GPRP_Optimisation.Method()

initialise_GPR_hyperparams -- Generates initial values for L-BFGF-S in GPy
call_GPy_optimise -- Returns optimised GPy model, given dataset, kernel, and optimisation bounds 
make_GPR_PER -- Returns L-BFGS-B optimised GPR predictions 
"""
from __future__ import division, print_function, absolute_import

import sys
import numpy as np
import GPy # Non standard Python package

from data_tools.load_raw_cluster_data import LoadExperiment as le
from gpr.common import get_data
from analysis_tools.common import sqr_err

class GPRPOptimisation(object):
    '''
    Returns GPy predictions using L-BFGF-B optimiser to fine tune each run
        test_case, variation -- Denotes scenario; use true noise as in LKFFB
        Sigma_Max -- Maximal bound for sigma in L-BFGS-B optimiser in GPy
        R_Max -- Maximal bound for R in L-BFGS-B optimiser in GPy

    
    '''
    def __init__(self, test_case, variation, 
                 Sigma_Max, R_Max, 
                 LKFFBfilepath, GPRP_savetopath):
        
        self.test_case = test_case 
        self.variation = variation  
        self.LKFFBfilepath = LKFFBfilepath 
        self.GPRP_savetopath = GPRP_savetopath

        self.dataobject = le(self.test_case, self.variation,
                             GPRP_load='No', 
                             LKFFB_load = 'Yes', LKFFB_path = self.LKFFBfilepath,
                             AKF_load='No',
                             LSF_load = 'No')

        self.Sigma_Max = Sigma_Max 
        self.R_Max = R_Max

        pass
    
    def initialise_GPR_hyperparams(self, approx_l_0=3.0):
        '''
        Returns initial values for L-BFGF-S in GPy

        SigmaMax, R_Max -- guessed manually; or via LKFFB Kalman values
                            GPy L-BFGS-B maxes out at boundaries.
                            SigmaMax, R_Max chosen such that predictions look sensible (manual tuning)
                            Further, Sigma_Max, R_Max are chosen s.t. p* from GPy is approx p_0, defined  above.
                            -- replace with a better approach--

        l_0 -- theoretically bounded on order of approx_l_0 * Delta_T_Sampling 
        Delta_T_Sampling -- as defined in analysis_tools.experiment
        p_0 -- theoretically bounded on order of n_train
        n_train -- as defined in analysis_tools.experiment
        '''
        # By theory-led approximations:
        p_0 = self.dataobject.Expt.n_train 
        l_0 = self.dataobject.Expt.Delta_T_Sampling * approx_l_0
        # By randomly chosen value between (0, max], where max == L-BFGFS Bound, tuned manually:
        # In the absence of msmt theory, we assume a uniform distribution over these parameters
        sigma_0 = np.random.uniform(low=0.1, high=self.Sigma_Max)
        R_0 = np.random.uniform(low=0.1, high=self.R_Max)
        return sigma_0, R_0, p_0, l_0
    
    
    def call_GPy_optimise(self, X, Y, sigma_0, R_0, p_0, l_0,
                          sigma_bound=0,
                          R_bound =0,
                          input_dim=1,
                          messages = False,
                          optimizer = None):
        ''' Returns optimised GPy model, given dataset, kernel, and optimisation bounds
        Choose lbfgs if required
        '''

        if sigma_bound ==0:
            sigma_bound = self.Sigma_Max

        if R_bound==0:
            R_bound = self.R_Max
        
        kernel_per = GPy.kern.StdPeriodic(input_dim, period=p_0, variance=sigma_0, lengthscale=l_0)
        gauss = GPy.likelihoods.Gaussian(variance=R_0)
        exact = GPy.inference.latent_function_inference.ExactGaussianInference()

        # intiate GPR model
        m1 = GPy.core.GP(X=X, Y=Y, kernel=kernel_per, likelihood=gauss, inference_method=exact)
        m1.std_periodic.variance.constrain_bounded(0, sigma_bound)
        m1.Gaussian_noise.variance.constrain_bounded(0, R_bound)

        # optimise GPR model
        #print('Before Optimisation: ', [m1.std_periodic.variance[0], m1.Gaussian_noise.variance[0], m1.std_periodic.period[0], m1.std_periodic.lengthscale[0]])
        m1.optimize(optimizer=optimizer, messages=messages) # Defaults to preferred optimiser if optimizer=None, preferred optimiser == lbfgsb
        #print('After Optimisation: ', [m1.std_periodic.variance[0], m1.Gaussian_noise.variance[0], m1.std_periodic.period[0], m1.std_periodic.lengthscale[0]])
        
        return m1
    
    
    def one_GPRP_model(self, training_pts, approx_l_0, randdata, messages=False, optimizer=None):
        '''
        Returns GPRP predictions for one truth, dataset, and GPRP initialisation
        '''
        

        X, Y, testx, truth, msmts  = get_data(self.dataobject, 
                                              points=training_pts,
                                              randomize=randdata)
        
        sigma_0, R_0, p_0, l_0 = self.initialise_GPR_hyperparams(approx_l_0=approx_l_0)

        init_params_list = [sigma_0, R_0, p_0, l_0]
        m1 = self.call_GPy_optimise(X, Y, sigma_0, R_0, p_0, l_0, messages=messages, optimizer=optimizer)
        opt_params_list = [m1.std_periodic.variance[0], m1.Gaussian_noise.variance[0], m1.std_periodic.period[0], m1.std_periodic.lengthscale[0]]
        
        predictions = m1.predict(testx)[0].flatten()
        
        return predictions, truth, msmts, opt_params_list, init_params_list, m1
        
        
    def make_GPR_PER(self, mapname='_GPR_PER_', approx_l_0=3.0, randdata='y'):
        ''' Returns GPRP predictions dataset for ensemble runs
        '''

        path2dir = self.GPRP_savetopath+'test_case_'+str(self.test_case)+'_var_'+str(self.variation)
        training_pts = self.dataobject.Expt.n_train # comparable in size to n_train, but training points are randomly chosen

        prediction_errors = [] 
        forecastng_errors = []
        macro_truth = []
        macro_data = []
        macro_opt_params = []
        macro_init_params = []

        for idx_d in xrange(self.dataobject.LKFFB_max_it_BR):

            predictions, truth, msmts, opt_params_list, init_params_list = self.one_GPRP_model(training_pts, approx_l_0, randdata)[0:5]

            truth_ = truth[self.dataobject.Expt.n_train - self.dataobject.Expt.n_testbefore : self.dataobject.Expt.n_train + self.dataobject.Expt.n_predict]
            residuals_sqr_errors = sqr_err(predictions, truth_)

            prediction_errors.append(residuals_sqr_errors[0: self.dataobject.Expt.n_testbefore])
            forecastng_errors.append(residuals_sqr_errors[self.dataobject.Expt.n_testbefore : ])
            macro_truth.append(truth)
            macro_data.append(msmts)
            macro_opt_params.append(opt_params_list)
            macro_init_params.append(init_params_list)

            np.savez(path2dir+'_GPR_PER_',
                msmt_noise_variance= self.dataobject.LKFFB_msmt_noise_variance, 
                max_it_BR = self.dataobject.LKFFB_max_it_BR, 
                macro_truth= macro_truth, 
                GPR_opt_params= macro_opt_params,
                macro_data = macro_data,
                GPR_init_params = macro_init_params,
                GPR_PER_prediction_errors=prediction_errors, 
                GPR_PER_forecastng_errors=forecastng_errors,
                training_pts=training_pts,
                Sigma_Max = self.Sigma_Max,
                R_Max=self.R_Max)

        return