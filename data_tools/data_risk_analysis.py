#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 19:20:43 2017
@author: riddhisw

.. module:: data_tools.data_risk_analysis

    :synopsis: Implements Bayes Risk analysis for all algorithms.

    Module Level Functions:
    ----------------------
        build_risk_dict : Return a dictionary data container for all algorithms
            reporting truths, prediction errors, hyperparameters data.
        norm_risk : Returns normalised Bayes Risk --
            (Risk predicted via algorithm) / (Risk predicted noise mean).
        sort_my_vals : Return ordered indices to sort input sequence from lowest to highest.
            [Helper Function.]
        kalman_risk : Return Kalman Risk by computing mean true error over truths, datasets.
            [Helper Function.]
        analyse_kalman_errs : Return Kalman (sigma, R) from lowest to highest Bayes Risk.
            [Helper Function.]
        riskmapdata : Return Kalman Bayes Risk Map data [for plotting]

.. moduleauthor:: Riddhi Gupta <riddhi.sw@gmail.com>

"""

from __future__ import division, print_function, absolute_import
import numpy as np
from analysis_tools.common import sqr_err

def build_risk_dict(LoadExperimentObject):
    '''
    HELPER FUNCTION

    Return a dictionary data container for all algorithms reporting:

    [0] State estimation error, where state estimation, n < N_train.
    [1] Hyper parameters for each error set.
    [2] Forecasting errors, where forecasting,  n > N_train.
    [3] True stochastic signals for each corressponding sets of errors.
    [4] Yes/No Flag for distinguishing LSF algorithm from KF and GPR algorithms.
    '''

    print('I am in RISK DICT and test case and variations are:', LoadExperimentObject.test_case, LoadExperimentObject.variation)

    RISKDICT = {}
    RISKDICT['LKFFB'] = [0, 0, 0, 0, 0]
    RISKDICT['AKF'] = [0, 0, 0, 0, 0]
    RISKDICT['GPRP'] = [0, 0, 0, 0, 0]
    RISKDICT['LSF'] = [0, 0, 0, 0, 0]
    RISKDICT['QKF'] = [0, 0, 0, 0, 0]

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

    if LoadExperimentObject.QKF_load == 'Yes':

        RISKDICT['QKF'][0] = LoadExperimentObject.QKF_macro_prediction_errors
        RISKDICT['QKF'][1] = LoadExperimentObject.QKF_random_hyperparams_list
        RISKDICT['QKF'][2] = LoadExperimentObject.QKF_macro_forecastng_errors
        RISKDICT['QKF'][3] = LoadExperimentObject.QKF_macro_truth
        RISKDICT['QKF'][4] = 'No'

    if LoadExperimentObject.GPRP_load == 'Yes':
        # np.newaxis creates a dummy axis to match (sigma, R) hypermeter choices in axis[0] of all AKF, LKFFB data

        RISKDICT['GPRP'][0] = LoadExperimentObject.GPRP_GPR_PER_prediction_errors[np.newaxis, ...]
        RISKDICT['GPRP'][1] = [None] # hyper-parameters optimised by GPy during analysis; or manually set priori to analysis
        RISKDICT['GPRP'][2] = LoadExperimentObject.GPRP_GPR_PER_forecastng_errors[np.newaxis, ...]
        RISKDICT['GPRP'][3] = LoadExperimentObject.GPRP_macro_truth[np.newaxis, ...]
        RISKDICT['GPRP'][4] = 'No'

    if LoadExperimentObject.LSF_load == 'Yes':
        # np.newaxis creates a dummy axis to match (sigma, R) hypermeter choices in axis[0] of all AKF, LKFFB data
        n_train = LoadExperimentObject.LSF_n_train
        lsf_f_err = LoadExperimentObject.LSF_macro_predictions[...,0][np.newaxis, ...]
        max_stps = lsf_f_err.shape[2]
        lsf_truth = LoadExperimentObject.LSF_macro_truths[np.newaxis, ..., n_train: n_train + max_stps]

        RISKDICT['LSF'][0] = np.zeros((1,1,1)) # no state estimation errors for LSF
        RISKDICT['LSF'][1] = [None] # hyper-parameters alpha_0, q, stp_fwd set prior to analysis
        RISKDICT['LSF'][2] = sqr_err(lsf_f_err, lsf_truth)
        RISKDICT['LSF'][3] = LoadExperimentObject.LSF_macro_truths[np.newaxis, ...]
        RISKDICT['LSF'][4] = 'Yes'

    return RISKDICT



#############################
# NORMALISED OPTIMISED RISK
#############################


def norm_risk(opt_fore_err,
               opt_truths,
               n_train,
               opt_state_err=0,
               LSF='No'):
    ''' Return Bayes Risk (predict via algorithm) / Bayes Risk (predict noise mean)

    Parameters:
    ----------
        opt_fore_err (`float64`) : Forecasting errors for tuned run.
            [dims: ensemble x maxforecaststeps, type=float64]
        opt_truths (`float64`) : Truth signal for opt_fore_err
            [dims: ensemble x number_of_points, type=float64]
        opt_state_err (`float64`) : State estimation errors for tuned run.
            (not applicable for LSF) [dims: ensemble x n_testbefore, type=float64]
        n_train (`int`) : Time step at which forecasting begins (training ends).
        ensemble (`int`) :  Number of trials over truths, datasets.
        prd_zero (`float64`) : Error from predicting the mean (==0) of any
            true noise.
    Note:
    n_testbefore == maxstateeststps
    n_predict >= maxforecaststeps


    Return:
    ------
        For AKF, LKFFB, GPR:
            opt_s_traj (`float64`) : Min. state estimation Bayes Risk for optimally tuned parameters.
            opt_f_traj (`float64`) : Min. forecasting / prediction Bayes Risk for optimally tuned parameters.

        For LSF:
            opt_f_traj (`float64`) : Min. forecasting / prediction Bayes Risk for optimally tuned parameters.

    '''

    prd_zero = np.mean(opt_truths**2, axis=0)
    maxforecaststeps = opt_fore_err.shape[1]

    opt_f_traj = np.mean(opt_fore_err, axis=0)/prd_zero[n_train : n_train + maxforecaststeps]

    if LSF == 'No' and opt_state_err.shape[0] != 1:

        # Test Pts < n_train is enabled for AKF, LKFFB (state estimates) and GPRP (interpolation)
        maxstateeststps = opt_state_err.shape[1]
        opt_s_traj = np.mean(opt_state_err, axis=0)/prd_zero[n_train - maxstateeststps : n_train]
        return opt_s_traj, opt_f_traj

    return None, opt_f_traj


############################################
# RISK MAP AND TRAJECTORIES (SIGMA V R)
############################################

def sort_my_vals(list_of_vals):
    ''' Return ordered indices to sort input sequence from lowest to highest.

    [Helper Function] - return original indices of list_of_vals in a sequence
    that reorders elements of list_of_vals from lowest to highest.
    '''
    indexed_vals = list(enumerate(list_of_vals))
    indices, sorted_vals = zip(*sorted(indexed_vals, key=lambda x: x[1]))

    return indices, sorted_vals

def kalman_risk(errs):
    ''' Returns Kalman Risk by computing mean true error over truths
    datasets. [Helper Function.]

    Parameters:
    ----------
        errs (`float64`) - Array of true errors, with dims* = [num_randparams x ensemble x n]
            Axis 0 : stores trials of hyperparmeters (sigma, R)
            Axis 1 : stores trials over truths and msmt noise (beta_z, msmts)
            Axis 2 : stores true error at different timesteps

        * Note:
            num_randparams (`int`) : Number of trials of (sigma, R)
            ensemble (`int`) : Number of trials over truths, datasets
            n (`int`) : num of timesteps under analysis (e.g. n_testbefore, n_predict)
    Returns:
    -------
        Mean true error, with expectation taken over truths and datasets.
            [DIM:  num_randparams x n, type=int]
    '''
    return np.mean(errs, axis=1)


def analyse_kalman_errs(errs, 
                        random_hyperparams_list,
                        maxstps):
    ''' Returns Kalman (sigma, R) from lowest to highest Bayes Risk.
            [Helper Function.]

    Parameters:
    ----------
        errs : Array of true errors. [DIM: num_randparams x ensemble x n, type=float64]
        random_hyperparams_list : Array of randomly chosen (sigma, R) [DIM: num_randparams x 2, type=float64]
        maxstps : Truncation of time_step under consideration (i.e. maxforecaststeps <= n_predict) [DIM: scalar, type=int]

    Returns:
    -------
    Return hyperparmeters (sigma, R) from random_hyperparams_list sorted in ascending.
        order of Bayes risk over errs and maxstps number of timesteps.

        indices : Index for (sigma, R) pairs sorted from lowest to highest Bayes Risk.
        srtd_risk : Sorted ascending values of Bayes Risk for each (sigma, R) pair.
        srtd_sigma : Sigma values corressponding to sorted risk values in srtd_risk.
        srtd_R : R values corressponding to sorted risk values in srtd_risk.
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
    ''' Return Kalman Bayes Risk Map data [for plotting].
    
    Parameters:
    ----------
         Takes as inputs objects in analysis_tools.riskanalysis.Bayes_Risk class:
            macro_state_err : Corressponds to Bayes_Risk.macro_prediction_errors.
            macro_fore_err : Corressponds to Bayes_Risk.macro_forecastng_errors.
            random_hyperparams_list : Corressponds to Bayes_Risk.random_hyperparams_list.
        maxforecaststps : Parameterises the number of time-steps used to define
            Bayes Risk metric during state estimation. Defaults to 1.
        maxstateeststps : Parameterises the number of time-steps used to define
            Bayes Risk metric during forecasting. Defaults to 1.
    
    Returns:
    -------
        Kalman Bayes Risk Map data for plotting:
            s_sigma (`float64`) : Values of sigma for lowest to highest total
                state estimation risk (s_risk).
            s_R (`float64`) : Values of R for lowest to highest total
                state estimation risk (s_risk).
            f_sigma (`float64`) : Values of sigma for lowest to highest total
                forecasting risk (f_risk).
            f_R (`float64`) : Values of R for lowest to highest total
                forecasting risk (f_risk).
            s_idx (`float64`) : Re-ordering of original index of (sigma, R) pairs
                corresponding to ascending s_risk.
            s_risk (`float64`) : Total state estimation risk (all timesteps) from
                lowest to highest in each (sigma, R) trial.
            s_traj (`float64`) : Trajectory of state estimation risk v. timesteps
                for ascending s_risk.
            f_traj (`float64`) : Trajectory of forecasting risk v. timesteps for
                ascending s_risk (not f_risk).
            f_idx (`float64`) : Re-ordering of original index of (sigma, R) pairs
                corresponding to ascending f_risk.
            f_risk (`float64`) : Total forecasting risk  (all timesteps) from lowest
                to highest in each (sigma, R) trial.
    '''
    s_idx, s_risk, s_sigma, s_R = analyse_kalman_errs(macro_state_err, random_hyperparams_list, maxstateeststps)
    f_idx, f_risk, f_sigma, f_R = analyse_kalman_errs(macro_fore_err, random_hyperparams_list, maxforecaststps)

    # Sort state estimation and forecasting risk trajectories in all regions based on
    # low loss (sigma, R) during training / state estimation. Mean taken along Axis 0 == mean over datasets
    s_traj = [np.mean(macro_state_err[x, :, :], axis=0) for x in s_idx]
    f_traj = [np.mean(macro_fore_err[x, :, :], axis=0)  for x in s_idx]

    return s_sigma, s_R, f_sigma, f_R, s_idx, s_risk, s_traj, f_traj, f_idx, f_risk
