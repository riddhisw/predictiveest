'''
Created on Thu Apr 20 19:20:43 2017
@author: riddhisw

.. module:: ls.LSF_risk_analysis

    :synopsis: Tunes LSF to appropriate alpha_0 value for LSF algorithm; 
        generates predictions.
    (statePredictions.py)

    Module Level Classes:
    ----------------------
        LSF_Optimisation :  Obtain predictions from a fully tuned LSF algorithm.

.. moduleauthor:: Riddhi Gupta <riddhi.sw@gmail.com>
'''
from __future__ import division, print_function, absolute_import

import sys
import numpy as np
sys.path.append('../')

from ls import state_predictions as sp

from data_tools.load_raw_cluster_data import LoadExperiment as le
from data_tools.data_risk_analysis import sort_my_vals
from ls.common import doLSF_forecast
from data_tools.common import get_data


class LSF_Optimisation(object):
    ''' Obtain predictions from a fully tuned LSF algorithm.

    Attributes:
    ----------
        test_case (`int`) : Index value to label parameter regimes, as
            in KalmanParameterRegimes.ods.
        variation (`int`) : Index value to label scanning parameter in a global
            fixed parameter regime, as in KalmanParameterRegimes.ods.
        LSF_past_msmts (`int`) : Number of past measurements to use in LSF model.
        LSF_steps_forward (`int`) : Number of steps ahead to predict in LSF model.
        datapath (`str`) : Path to file for LKFFB database of engineered true
            realisations of dephasing field.
        LSF_steps_between_msmts (`int`, optional) : Number time-steps between measurements.
                Defaults to 1.
        LSF_err_train_iter | LSF_iter (`int`, optional) : Maximum number of iterations
            for gradient descent to search for AR(q) weights. Argument is passed
            on as `numIters` in statePredictions_2.gradient_descent.
        LSF_alpha_iter (`int`) : Defaults to 50. [DEPRECIATED]
        LSF_ensembl_size (`int`, optional) : Number of different noise realisations in
            ensemble for LSF analysis. Defaults to 50.

    Methods:
    -------
        loss_lsf : Return loss value for alpha_0 "optimisation".
        train_alpha0_manual : Returns optimal alpha_0 based on lowest loss from a
             set of candidates.
        make_LS_Ensemble_data : Returns LSF predictions using distribution in both
            training and validation datasets.
        get_trained_weights_dist : Returns distribution of weights for distribution
            in training datasets.
    '''

    def __init__(self, test_case, variation, LSF_past_msmts, LSF_steps_forward, datapath,
                LSF_steps_between_msmts=1, LSF_iter=50, LSF_alpha_iter=50, ensembl_size=50):

        self.dataobject = le(test_case, variation,
                             GPRP_load='No',
                             LKFFB_load='Yes',
                             LKFFB_path=datapath,
                             AKF_load='No',
                             LSF_load='No')
        self.test_case = test_case
        self.variation = variation
        self.LSF_past_msmts = LSF_past_msmts
        self.LSF_steps_forward = LSF_steps_forward
        self.datapath = datapath
        self.LSF_steps_between_msmts = LSF_steps_between_msmts
        self.LSF_err_train_iter = LSF_iter
        self.LSF_ensembl_size = ensembl_size


    def loss_lsf(self, try_alpha0, user_msmt_train=0):

        '''Return error train and AR(q) weights from training LSF, and tuning
        gradient descent hyperparameters, alpha_0, based min total loss.

        Parameters:
        ----------
            try_alpha0 (`float64`) : Hyper-parameter in gradient descent in LSF.
            user_msmt_train (`float64`) : Specifies training data for LSF:
                If user_msmt_train != 0:
                    Gradient descent in LSF will use one data set for all training.
                If user_msmt_train == 0:
                    A new noise realisation and msmt data is generated for each
                    new choice of try_alpha0 i.e. loss_lsf == Bayes Risk.
                Defaults to 0.

        Returns:
        -------
            errorTrain (`float64`): Residuals at each iteratiion step of gradient
                descent in LSF.
            weights (`float64`): Weights learned from training data (`q` weights for
                an approximation to an autoregressive process AR(q)).
        '''

        if np.sum(user_msmt_train) != 0:
            measurements_train = user_msmt_train

        elif np.sum(user_msmt_train) == 0:
            measurements_train = get_data(self.dataobject)[0]

        training_data = sp.build_training_dataset(measurements_train,
                                                  past_msmts=self.LSF_past_msmts,
                                                  steps_forward=self.LSF_steps_forward,
                                                  steps_between_msmts=self.LSF_steps_between_msmts)

        weights, errorTrain = sp.gradient_descent(training_data, self.LSF_err_train_iter, alpha_coeff=try_alpha0)

        # Based on the last value of err train (not gradient estimate of error train)
        lossval = errorTrain[-1] 

        if np.isfinite(lossval):
            return lossval, errorTrain, weights[:, 0]
        print("Lossval not finite")
        return None, errorTrain, weights[:, 0]


    def train_alpha0_manual(self, arr_alphas):

        '''Return lowest loss alpha_0 from arr_alphas.

        Parameters:
        ----------

            arr_alphas (`float64`): Set of candidate alpha values.BaseException

        Returns:
        -------
            arr_alphas[index[0]] (`float64`) : Optimally tuned (lowest loss) alpha value.
            index (`float64`) : Index (in original array) corressping to optimal alpha.
            lossvalTrain (`float64`) : Representive loss value for one run of LSF
                with a given alpha.
            errTrains (`float64`) : Residual error for each iterative step within
                one run of LSF, for all alphas.
            weightTrain (`float64`) : AR(q) weights discovered by  LSF for all values of alpha.
        '''

        iter_ = arr_alphas.shape[0]
        lossvalTrain = np.zeros(iter_)
        errTrains = np.zeros((iter_, self.LSF_err_train_iter))
        weightTrain = np.zeros((iter_, self.LSF_past_msmts))

        # use only one dataset for training
        measurements_train = get_data(self.dataobject)[0]

        for idx in xrange(iter_):
           lossvalTrain[idx], errTrains[idx, :], weightTrain[idx, :] = self.loss_lsf(arr_alphas[idx], user_msmt_train=measurements_train)

        # pick alpha in arr_alpha with the lowest loss
        # pairs = zip(arr_alphas, lossvalTrain)
        # alpha_srtd, loss_srtd = zip(*sorted(pairs, key=(lambda x: x[1])))

        index, losses = sort_my_vals(lossvalTrain)
        return arr_alphas[index[0]], index, lossvalTrain, errTrains, weightTrain



    def make_LS_Ensemble_data(self, pick_alpha0, savetopath, num_of_iterGD=50):
        '''
        Saves LSF predictions analysis as an .npz file. 

        Parameters:
        ----------
             pick_alpha0 (`float64`) : Hyper-parameter for gradient descent tuning.
             savetopath (`str`) : Filepath for saving LSF analysis output as a .npz file.
             num_of_iterGD (`int`, optional) : Number of iterations of gradient descent in LSF.
                Defaults to 50.

        Returns:
        -------
            Saves LSF predictions analysis as an .npz file.
        '''

        n_train = self.dataobject.Expt.n_train
        n_start_at = n_train - self.LSF_past_msmts + 1

        # measurements_train, pick_train = get_data(self.dataobject) # implementation in DRAFT 1, DATA_v0/testingLSF/

        macro_weights = []
        macro_predictions = []
        macro_actuals = []
        macro_errorTrain_fore = [] # changes by steps forwards, not a risk avergae.
        macro_truths = []
        macro_data = []

        for idx_en in xrange(self.LSF_ensembl_size):


            # desired implementation in DATA v0
            measurements_train, pick_train = get_data(self.dataobject)

            measurements_val, pick_val = get_data(self.dataobject)
            shape = self.dataobject.LKFFB_macro_truth.shape
            noisetrace_val = self.dataobject.LKFFB_macro_truth.reshape(shape[0]*shape[1], shape[2])[pick_val, :]      

            output = doLSF_forecast(measurements_train,
                                    measurements_val,
                                    pick_alpha0, n_start_at,
                                    self.LSF_steps_forward,
                                    self.LSF_past_msmts,
                                    steps_between_msmts=self.LSF_steps_between_msmts,
                                    num_of_iterGD=num_of_iterGD)

            macro_weights.append(output[1])
            macro_predictions.append(output[2])
            macro_actuals.append(output[3])
            macro_errorTrain_fore.append(output[4])
            macro_truths.append(noisetrace_val)
            macro_data.append(measurements_val)
            # Save after each run
            np.savez(savetopath+'test_case_'+str(self.test_case)+'_var_'+str(self.variation)+'_LS_Ensemble',
                     n_start_at=n_start_at,
                     past_msmts=self.LSF_past_msmts,
                     n_train=n_train, ################################## n_train not used
                     n_predict=self.LSF_steps_forward,
                     test_case=self.test_case,
                     var=self.variation,
                     pick_alpha=pick_alpha0,
                     macro_weights=macro_weights,
                     macro_predictions=macro_predictions,
                     macro_actuals=macro_actuals,
                     macro_truths=macro_truths,
                     macro_data=macro_data,
                     # pick_train=pick_train,
                     # measurements_train=measurements_train,
                     macro_errorTrain_fore=macro_errorTrain_fore)


    def get_trained_weights_dist(self, pick_alpha0, iter_=50):

        '''Returns data matrices for iter runs of LSF using pick_alpha0.
        Typically set pick_alpha0 as optimal alpha after LSF tuning, and see
        the distribution of AR weights driven by underlying true noise
        realisations.

        Parameters:
        ----------
            pick_alpha0 (`float64`) : Alpha (hyper-parameter) in gradient descent in LSF.
            iter_ (`int`, optional) : Number of total true dephasing noise realisations.
                Defaults to 50.

        Returns:
        -------
            macro_err_trains (`float64`) : Set of residual sequences for each
                true realisation in an ensemble of iter_ total realisations.
            macro_weights (`float64`) : Set of AR weights for each
                true realisation in an ensemble of iter_ total realisations.

        '''

        macro_err_trains = np.zeros((iter_, self.LSF_err_train_iter))
        macro_weights = np.zeros((iter_, self.LSF_past_msmts))

        for idx_run in xrange(iter_):
            macro_err_trains[idx_run, :], macro_weights[idx_run, :] = self.loss_lsf(pick_alpha0)[1:]

        return macro_err_trains, macro_weights
