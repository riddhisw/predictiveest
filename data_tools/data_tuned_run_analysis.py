#!/usr/bin/env python2
# -*- coding: utf-8 -*-
'''
Created on Thu Apr 20 19:20:43 2017
@author: riddhisw

.. module:: data_tools.data_tuned_run_analysis

    :synopsis: Generates single predictions and spectrum estimates from tuned filters.

    Module Level Functions:
    ----------------------
        LSF_run : Return a single LSF prediction based on tuned hyper parameters.
        calc_AR_PSD : Return PSD S(w) estimate for covariance stationary
            autoregressive process, AR(q), of order q.
        AKF_run : Return a single AKF prediction and spectral density estimate.
        clip_high_periods : Return GPRP periods <= theory; returns original
            periods if theory bound cannot be met.
        choose_GPR_params : Return GPRP parameters using lowest mean sqr err
            and physically sensible periodicities
        GPRP_run : Return a single GPRP prediction based on tuned hyper parameters.
        LKFFB_amps : Return LKFFB spectral estimate and theoretical PSD.
        LKFFB_run : Return a single LKFFB prediction and spectral density estimate.

.. moduleauthor:: Riddhi Gupta <riddhi.sw@gmail.com>

'''

from __future__ import division, print_function, absolute_import
import numpy as np
import kf.fast_2 as Kalman


from kf.common import calc_inst_params
from akf.armakf import autokf as akf
from ls import statePredictions as sp
from analysis_tools.truth import SYMMETRIC_ONE_SIDED_PSD, LKFFB_HILBERT_TRANSFORM_

################
# LSF
################

def LSF_run(LdExp, y_signal):
    '''
    Return a single LSF prediction based on tuned hyper parameters.

    Parameters:
    ----------
        LdExp (`class object`) :  A data_tools.load_raw_cluster_data.LoadExperiment instance.
        y_signal (`float64`) : Noisy measurements (input for filtering).

    Returns:
    -------
        predictions (`float64`) : single LSF prediction sequence.
    '''

    if LdExp.LSF_load != 'No':

        mean_weights = np.mean(LdExp.LSF_macro_weights[:, :, :, 0], axis=0)
        # Truths could be randomly chosen for direct comparison with AKF.
        # Doesn't matter as variance is low for test_cases considered.

        order = mean_weights.shape[1]
        n_start_at = LdExp.LSF_n_start_at
        sequence_of_past_msmts_reordered = y_signal[n_start_at : n_start_at + order][::-1]
        predictions = np.dot(mean_weights, sequence_of_past_msmts_reordered)

        return predictions


################
# AKF
################

def calc_AR_PSD(phi_p, sigma_n_sqr, Delta_S_Sampling, Delta_T_Sampling):
    ''' Return power spectral density (PSD) S(w) estimate for covariance
        stationary autoregressive process, AR(q), of order q.

        Parameters:
        ----------
            phi_p (`float64`): learned autoregressive weights (i.e. from LSF) [dims: q x 1]
            sigma_n_sqr (`float64`): noise covariance (i.e. AKF Kalman's sigma) [scalar]
            Delta_S_Sampling (`float64`): Fourier resolution, defined in
                analysis_tools.experiment [scalar].
            Delta_T_Sampling (`float64`): time between msmts, defined in
                analysis_tools.experiment [scalar].

        Returns:
        -------
            omega (`float64`): Frequency axis in Fourier space, set by experiemnt.
            S_thr (`float64`): Spectrum estimate from learned AR coefficients.

        See Also:
        --------
            Ref 1. Lecture Notes (http://www.lmd.ens.fr/E2C2/class/SASP_lecture_notes.pdf)
            Ref 2. Wikipedia (https://en.wikipedia.org/wiki/Autoregressive_model#Spectrum)

    '''
    number_of_points = 1.0/(Delta_S_Sampling*Delta_T_Sampling)
    halfN = int((number_of_points - 1) / 2.0)
    order_p = phi_p.shape[0]

    S_thr = np.zeros(halfN)
    omega = np.zeros(halfN)

    for idx_j in range(1, halfN, 1):

        omega_j = 2.0*np.pi*idx_j*Delta_S_Sampling

        one_j_term = 0
        for idx_p in range(0, order_p, 1):
            one_j_term += phi_p[idx_p]*np.exp(-1.0j*(idx_p + 1)*Delta_T_Sampling*omega_j)

        S_thr[idx_j] = sigma_n_sqr/(abs(1 - one_j_term)**2)
        omega[idx_j] = omega_j

    return omega, S_thr


def AKF_run(LdExp, y_signal, **kwargs):
    '''Return a single AKF prediction and spectral density estimate.

    Parameters:
    ----------
        LdExp (`class object`) : A data_tools.load_raw_cluster_data.LoadExperiment instance.
        y_signal (`float64`) : Noisy measurements (input for filtering).

    Returns:
    -------
        akf_x (`float64`) : Frequency axis (radians) for reconstructed spectrum
            estimate, S(w) [dims: halfN x 1].
        akf_y (`float64`) :  S(w) estimate based on AR  weights [dims: halfN x 1]
        akf_y_norm (`float64`) :  S(w) estimate * (1.0/ sum (S(w) True))
        akf_pred (`float64`) : Predictions sequence from a single run of AKF
    '''
    if LdExp.AKF_load != 'No':

        oe = 0.0
        rk = 0.0

    try:
        quantised = kwargs['quantised']
        print('Got here')
    except:
        quantised = 'No'

    if len(kwargs) == 2:
        oe = kwargs['opt_sigma'] # Optimally tuned
        rk = kwargs['opt_R'] # Optimally tuned

    weights = LdExp.AKF_weights # This is randomly chosen. They are not ensmble averaged weights.
    order = weights.shape[0]

    akf_pred = akf('AKF', y_signal, weights, oe, rk,
                   n_train=LdExp.Expt.n_train,
                   n_testbefore=LdExp.Expt.n_testbefore,
                   n_predict=LdExp.Expt.n_predict,
                   p0=LdExp.LKFFB_kalman_params[3], # same as LKFFB p0
                   skip_msmts=1,
                   save='No', quantised=quantised)

    akf_x, akf_y = calc_AR_PSD(weights, oe, LdExp.Expt.Delta_S_Sampling, LdExp.Expt.Delta_T_Sampling)

    # Normalisation against Truth
    LdExp.Truth.beta_z_truePSD() # new line. If beta_z_truePSD() is not called, true_S_norm = None
    akf_y_norm = akf_y*1.0/LdExp.Truth.true_S_norm

    return akf_x, akf_y, akf_y_norm, akf_pred


################
# GPRP
################

def clip_high_periods(LdExp, ordered_idxp, ordered_periods, n_train_default=2000, threshold=1.1):
    '''
    Return GPRP periods <= theory; returns original periods if theory bound cannot be met.
    [Helper Function.]
    '''

    try:
        n_train = LdExp.Expt.n_train
    except:
        n_train = n_train_default
    print(" Theory value of GPRP periodicity implied in choose_GPR_params():", n_train)

    clip_high_periods = []

    for idx_pair in zip(ordered_idxp, ordered_periods):

        if idx_pair[1] <= n_train_default*threshold: # theoretically estimated bound on GPRP period
            clip_high_periods.append(idx_pair)

    idxp_, periods_ = zip(*clip_high_periods)

    if len(periods_) < 2:
        print("clip_high_periods() was unable to clip periods; original periods were returned")
        return ordered_idxp, ordered_periods

    return list(idxp_), list(periods_)


def choose_GPR_params(LdExp):
    '''
    Return GPRP parameters using lowest mean sqr err and physically sensible
        periodicities.

    Parameters:
    ----------
        LdExp -- A data_tools.load_raw_cluster_data.LoadExperiment instance.
        LdExp will access...
            LdExp.GPRP_GPR_PER_prediction_errors (`float64`) : Residual sqr error between
                predictions and truth [dims: maxit_BR x n_testbefore].
            LdExp.GPRP_GPR_opt_params[:, 2] (`float64`) : Tuned periods, as in Periodic Kernel
                using GPy L-BFGS-B trials [dims: maxit_BR x 1].
            maxit_BR (`int`): Number of trials using GPy's L-BFGS-B, given the same
                initial values for hyperpameters for all trials [DIMS: scalar].

    Returns:
    -------
        ans ('float64'): Set of GPRP parameters with the lowest mean square error
            and physically sensible periodicities.
    '''

    from data_tools.data_risk_analysis import sort_my_vals

    # Calculate loss:= mean sqr err over timesteps
    loss = np.mean(LdExp.GPRP_GPR_PER_prediction_errors, axis=1)

    # and sort losses from low to high over all  GPy trials
    idxs, vals = sort_my_vals(loss)

    # Get different periods used in GPy trials
    idxp_, periods_ = sort_my_vals(LdExp.GPRP_GPR_opt_params[:, 2])

    # Cut out runs with periods >>> n_train
    idxp, periods = clip_high_periods(LdExp, idxp_, periods_)

    # Total number of GPy L-GFBS-B trials - number of runs discarded by clip_high_periods()
    trials = LdExp.GPRP_GPR_PER_prediction_errors.shape[0]  + len(periods) - len(periods_)

    for increment in range(1, trials, 1):

        try:
            ok_losses = set(idxs[0:increment])
            ok_periods = set(idxp[trials-increment:])

            # Find low loss trials with sensible periods
            ideal_instances = list(ok_losses.intersection(ok_periods))

            if len(ideal_instances) >= 1 and increment <= 4:

                first_instance = ideal_instances[0]
                print('Let "Ideal Instances" == low prediction losses + theoretically sensible periodicities for some hyperparmeter set in GPRP')
                print('First Few Ideal Instances:', ideal_instances, ' in bottom (top) %s loss values (periodicity values)' %(increment))

            if len(ideal_instances) >= 10:

                print('Other Instances:', ideal_instances, ' in bottom (top) %s loss values (periodicity values)' %(increment))
                break
        except:
            print("No overlap found between lowest losses and sensible periods for GPRP")
            continue
    try:
        ans = LdExp.GPRP_GPR_opt_params[first_instance, :]
    except:
        ans = LdExp.GPRP_GPR_opt_params[ideal_instances[0], :]

    return ans


def GPRP_run(LdExp, y_signal):
    '''
    Return a single GPRP prediction based on tuned hyper parameters.

    Parameters:
    ----------
        LdExp (`class object`) : A data_tools.load_raw_cluster_data.LoadExperiment instance.
        y_signal (`float64`) : Noisy measurements (input for filtering).

    Returns:
    -------
        predictions (`float64`) : Predictions sequence from a single run of AKF.
    '''

    import GPy

    # Create training data objects and test pts for GPy
    X = LdExp.Expt.Time_Axis[0:LdExp.Expt.n_train, np.newaxis]
    Y = y_signal[0:LdExp.Expt.n_train, np.newaxis]
    testx = LdExp.Expt.Time_Axis[LdExp.Expt.n_train - LdExp.Expt.n_testbefore : ]

    #Set Chosen Params for GPR Model
    if LdExp.GPRP_load != 'No':

        R, sigma, period, length_scale = choose_GPR_params(LdExp) # pick tuned parameters

        kernel_per = GPy.kern.StdPeriodic(1, period=period, variance=sigma, lengthscale=length_scale)
        gauss = GPy.likelihoods.Gaussian(variance=R)
        exact = GPy.inference.latent_function_inference.ExactGaussianInference()
        m1 = GPy.core.GP(X=X, Y=Y, kernel=kernel_per, likelihood=gauss, inference_method=exact)

        # Predict
        predictions = m1.predict(testx[:,np.newaxis])[0].flatten()

        return predictions

################
# LKFFB
################

def LKFFB_amps(LdExp, freq_basis_array=None, instantA=None):
    '''
    Return LKFFB spectral estimate and theoretical PSD.

    Parameters:
    ----------
        LdExp (`class object`) : A data_tools.load_raw_cluster_data.LoadExperiment instance.
        freq_basis_array (`float64`) : Fixed basis for LKFFB, as defined in kf.fast_2.
        instantA (`float64`) : Learned instantaneous amplitudes in LKFFB,
            as defined in kf.fast_2.

    Returns:
    --------
        x_data (`float`): List of Fourier frequencies [LKFFB, theory].
        y_data (`float`): List of Fourier spectrum values [LKFFB amplitudes, theory].
        Output[2] (`float`): Total energy [LKFFB, theory].
    '''


    LdExp.Truth.beta_z_truePSD()

    if (freq_basis_array == None) and (instantA == None):
        freq_basis_array = LdExp.Truth.freq_basis_array
        instantA = LdExp.Truth.instantA

    x_data = [2.0*np.pi*freq_basis_array, LdExp.Truth.true_w_axis[LdExp.Truth.J -1:]]

    kalman_amps = (instantA**2)*(2*np.pi)*LKFFB_HILBERT_TRANSFORM_ 
    theory_PSD = SYMMETRIC_ONE_SIDED_PSD*LdExp.Truth.true_S_twosided[LdExp.Truth.J -1:]

    norm_kalman_amps = kalman_amps*(1.0/ LdExp.Truth.true_S_norm)
    norm_theory_PSD = theory_PSD*(1.0/ LdExp.Truth.true_S_norm)

    y_data = [norm_kalman_amps, norm_theory_PSD]

    return x_data, y_data, [np.sum(kalman_amps), LdExp.Truth.true_S_norm] 
    # why are we not returning normalised total power?

def LKFFB_run(LdExp, y_signal, **kwargs):
    '''
    Return a single LKFFB prediction and spectral density estimate.

    Returns:
    -------
        predictions (`float64`): Predictons from a single run of LKFFB
            [dims: n_testbefore + n_predict x 1].
        x_data[0] (`float64`): Omega axis for LKFFB using freq_basis_array [radians],
            [DIMS: numf x 1].
        x_data[1] (`float64`): Omega axis for True S(w) [radians], [dims: J x 1].
        y_data[0] (`float64`): S(w) estimate based on LKFFB instantaneous
            amplitudes * (1.0/ sum (S(w) True)), [dims: numf x 1].
        y_data[1] (`float64`): True S(w) * (1.0/ sum (S(w) True)), [dims: J x 1].
        true_S_norm[0] (`float64`): sum (S(w) LKFFB), [dims: scalar].
        true_S_norm[1] (`float64`): sum (S(w) True), [dims: scalar].

    Parameters:
    ----------
        LdExp (`class object`) :  A data_tools.load_raw_cluster_data.LoadExperiment instance.
        y_signal (`float64`) : Noisy measurements (input for filtering).
        **kwargs :
            kwargs['opt_sigma'] : Optimally tuned process noise variance for kf.fast_2.
            kwargs['opt_R'] : Optimally tuned measurement noise variance for kf.fast_2.

    Harcoded Fixed Parameters:
    -------------------------
        method (`int`): 'ZeroGain', a type of prediction method defined in kf.fast_2
        freq_basis_array (`float64`): Basis A, a type of fixed basis defined in kf.fast_2
    '''
    oe = 0.0
    rk = 0.0

    # FIXED LKFFB PARAMETER: Prediction Method 'ZeroGain'
    method = 'ZeroGain'

    try:
        quantised = kwargs['quantised']
    except:
        quantised = 'No'

    if len(kwargs) == 2:

        # Optimally tuned Kalman Process noise
        oe = kwargs['opt_sigma']

        # Optimally tuned Kalman Measurement noise
        rk = kwargs['opt_R']

    x0 = LdExp.LKFFB_kalman_params[2]
    p0 = LdExp.LKFFB_kalman_params[3]
    bdelta = LdExp.LKFFB_kalman_params[4]

    # FIXED LKFFB PARAMETER: Choose Basis A
    freq_basis_array = np.arange(0.0, LdExp.Expt.bandwidth, bdelta)

    predictions, x_hat = Kalman.kf_2017(y_signal,
                                        LdExp.Expt.n_train,
                                        LdExp.Expt.n_testbefore, LdExp.Expt.n_predict,
                                        LdExp.Expt.Delta_T_Sampling,
                                        x0, p0, oe, rk, freq_basis_array,
                                        phase_correction=0,
                                        prediction_method=method,
                                        skip_msmts=1, switch_off_save='Yes', quantised=quantised)

    x_hat_slice = x_hat[:, :, LdExp.Expt.n_train]
    instantA, instantP = calc_inst_params(x_hat_slice)

    x_data, y_data, LdExp.Truth.true_S_norm = LKFFB_amps(LdExp,
                                                         freq_basis_array=freq_basis_array,
                                                         instantA=instantA)
    # LKFFB and Theory Data (List form: [LKFFB, Theory])
    return x_data, y_data, LdExp.Truth.true_S_norm, predictions

TUNED_RUNS_DICT = {}
TUNED_RUNS_DICT['LSF'] = LSF_run
TUNED_RUNS_DICT['AKF'] = AKF_run
TUNED_RUNS_DICT['GPRP'] = GPRP_run
TUNED_RUNS_DICT['LKFFB'] = LKFFB_run
