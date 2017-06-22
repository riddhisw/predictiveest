# Helper Functions for Analysis
import numpy as np

##############
# STANDARDIZE DATA FOR RISK ANALYSIS
###############

def build_risk_dict(LoadExperimentObject):
    
    from analysis_tools.common import sqr_err

    print('I am in RISK DICT and test case and variations are:', LoadExperimentObject.test_case, LoadExperimentObject.variation)
    
    RISKDICT={}
    RISKDICT['LKFFB'] = [0, 0, 0, 0, 0]
    RISKDICT['AKF'] = [0, 0, 0, 0, 0]
    RISKDICT['GPRP'] = [0, 0, 0, 0, 0]
    RISKDICT['LSF']= [0, 0, 0, 0, 0] 
    
    
    if LoadExperimentObject.LKFFB_load == 'Yes':
    
        RISKDICT['LKFFB'][0] = LoadExperimentObject.LKFFB_macro_prediction_errors
        RISKDICT['LKFFB'][1] = LoadExperimentObject.LKFFB_random_hyperparams_list
        RISKDICT['LKFFB'][2] = LoadExperimentObject.LKFFB_macro_forecastng_errors
        RISKDICT['LKFFB'][3] = LoadExperimentObject.LKFFB_macro_truth
        RISKDICT['LKFFB'][4] = 'No'
        
        
    if LoadExperimentObject.AKF_load == 'Yes':
        
        RISKDICT['AKF'][0] = LoadExperimentObject.AKF_akf_macro_prediction_errors
        RISKDICT['AKF'][1] = LoadExperimentObject.AKF_random_hyperparams_list
        RISKDICT['AKF'][2] = LoadExperimentObject.AKF_akf_macro_forecastng_errors
        RISKDICT['AKF'][3] = LoadExperimentObject.AKF_macro_truth
        RISKDICT['AKF'][4] = 'No'

        
    if LoadExperimentObject.GPRP_load == 'Yes':
        
        RISKDICT['GPRP'][0] = LoadExperimentObject.GPRP_GPR_PER_prediction_errors[np.newaxis, ...]
        RISKDICT['GPRP'][1] = [None]
        RISKDICT['GPRP'][2] = LoadExperimentObject.GPRP_GPR_PER_forecastng_errors[np.newaxis, ...]
        RISKDICT['GPRP'][3] = LoadExperimentObject.GPRP_macro_truth[np.newaxis, ...]
        RISKDICT['GPRP'][4] = 'No'

        
    if LoadExperimentObject.LSF_load == 'Yes':
        
        n_train = LoadExperimentObject.Expt.n_train
        lsf_f_err = LoadExperimentObject.LSF_macro_predictions[...,0][np.newaxis, ...]
        max_stps = lsf_f_err.shape[2]
        lsf_truth = LoadExperimentObject.LSF_macro_truths[np.newaxis, ..., n_train: n_train + max_stps]
        
        RISKDICT['LSF'][0] = np.zeros((1,1,1))
        RISKDICT['LSF'][1] = [None]
        RISKDICT['LSF'][2] = sqr_err(lsf_f_err, lsf_truth)
        RISKDICT['LSF'][3] = LoadExperimentObject.LSF_macro_truths[np.newaxis, ...]
        RISKDICT['LSF'][4] = 'Yes'
    
    return RISKDICT



##############
# NORMALISED OPTIMISED RISK
###############


def norm_risk(opt_fore_err, 
               opt_truths, 
               n_train,
               opt_state_err=0,
               LSF='No'):
    '''Returns:
    opt_s_traj -- s_traj[0] Dims: ensmble x n_testbefore (== maxstateeststps)
    opt_f_traj -- f_traj[0] Dims: ensmble x n_predict (>= maxforecaststeps)
    '''       
    
    prd_zero = np.mean(opt_truths**2, axis=0)
    maxforecaststeps = opt_fore_err.shape[1]

    opt_f_traj = np.mean(opt_fore_err, axis=0)/prd_zero[n_train : n_train + maxforecaststeps ]
    
    if LSF =='No' and opt_state_err.shape[0] !=1:
        
        # Test Pts < n_train is enabled for AKF, LKFFB (state estimates) and GPRP (interpolation)
        
        maxstateeststps = opt_state_err.shape[1]
        opt_s_traj = np.mean(opt_state_err, axis=0)/prd_zero[n_train - maxstateeststps : n_train ]
        return opt_s_traj, opt_f_traj
    
    return None, opt_f_traj


##############
# RISK MAP AND TRAJECTORIES (SIGMA V R)
###############

def sort_my_vals(list_of_vals):
    '''
    Returns:
    indices: returns original indices of list_of_vals after re-ordering
    sorted_vals: list_of_vals ordered from lowest to highest
    '''
    indexed_vals = list(enumerate(list_of_vals))
    indices, sorted_vals = zip(*sorted(indexed_vals, key= lambda x: x[1]))
    
    return indices, sorted_vals

def kalman_risk(errs):
    '''
    errs - Array of true errors. Axis 0 stores trials of hyperparmeters 
    (sigma, R). Axis 1 stores trials over truths and msmt noise. Axis 2 
    stores true error at different timesteps. 
    
    Returns: 2D Kalman Risk Array. Axis 0 stores risk over hyperparmeter trials; 
    Axis 1 stores risk over timestep.
    '''
    return np.mean(errs, axis=1) 


def analyse_kalman_errs(errs, 
                        random_hyperparams_list,
                        maxstps):
    '''
    errs - Array of true errors. Axis 0 stores trials of random 
    (sigma, R) in random_hyperparams_list. Axis 1 stores trials 
    over truths and msmt noise. Axis 2 stores true 
    error at different timesteps. 
    
    Returns: hyperparmeters sigma, R from random_hyperparams_list sorted in ascending 
    order based on risk calculation over errs over maxstps number of timesteps
    
    '''
    # total_risk == mean of kalman_risk over timesteps
    # not total_risk == sum of kalman_risk over timesteps
    total_risk = np.mean(kalman_risk(errs[:, :, 0 : maxstps]), axis=1) 
    
    indices, srtd_risk = sort_my_vals(total_risk)

    # Sorts hyper parameters in ascending order of srtd_risk
    srtd_sigma, srtd_R = np.asarray(zip(*[random_hyperparams_list[x] for x in indices]))
    
    return indices, srtd_risk, srtd_sigma, srtd_R


def riskmapdata(macro_state_err, 
                macro_fore_err, 
                random_hyperparams_list,
                maxforecaststps=1,
                maxstateeststps=1):
        '''Returns:
        s_sigma -- Values of sigma for lowest to highest total state estimation risk (s_risk)
        s_R -- Values of R for lowest to highest total state estimation risk (s_risk)
        
        f_sigma -- Values of sigma for lowest to highest total forecasting risk (f_risk)
        f_R -- Values of R for lowest to highest total forecasting risk (f_risk)
        
        s_idx -- Re-ordering of original index of (sigma, R) pairs corresponding to ascending s_risk 
        s_risk -- Total state estimation risk  (all timesteps) from lowest to highest in each (sigma, R) trial
        
        s_traj -- Trajectory of state estimation risk v. timesteps for ascending s_risk
        f_traj -- Trajectory of forecasting risk v. timesteps for ascending s_risk (not f_risk)
        '''
        s_idx, s_risk, s_sigma, s_R = analyse_kalman_errs(macro_state_err, random_hyperparams_list, maxstateeststps)
        f_idx, f_risk, f_sigma, f_R = analyse_kalman_errs(macro_fore_err, random_hyperparams_list, maxforecaststps)
        
        # Sort state estimation and forecasting risk trajectories in all regions based on 
        # low loss (sigma, R) during training / state estimation. Mean taken along Axis 0 == mean over datasets
        s_traj = [np.mean(macro_state_err[x, :, :], axis=0) for x in s_idx]
        f_traj = [np.mean(macro_fore_err[x, :, :], axis=0)  for x in s_idx] 
                
        return s_sigma, s_R, f_sigma, f_R, s_idx, s_risk, s_traj, f_traj, f_idx, f_risk
