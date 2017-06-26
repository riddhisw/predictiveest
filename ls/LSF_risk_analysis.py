'''
Created on Thu Apr 20 19:20:43 2017

@author: riddhisw

PACKAGE: ls
MODULE: LSF_risk_analysis
PURPOSE: Tunes LSF to appropriate alpha_0 value for LSF algorithm; generates predictions

CLASSES: LSF_Optimisation
METHODS: LSF_Optimisation.methods()

loss_lsf -- Returns loss value for alpha_0 "optimisation"
train_alpha0_manual -- Returns optimal alpha_0 based on lowest loss from a set of candidates
make_LS_Ensemble_data -- Returns LSF predictions using distribution in both training and validation datasets
get_trained_weights_dist -- Returns distribution of weights for distribution in training datasets
'''

from __future__ import division, print_function, absolute_import

import sys
import numpy as np
sys.path.append('../')

from ls import statePredictions as sp

from data_tools.load_raw_cluster_data import LoadExperiment as le
from data_tools.data_risk_analysis import sort_my_vals
from ls.common import doLSF_forecast
from data_tools.common import get_data


class LSF_Optimisation(object):
    

    def __init__(self, test_case, variation, LSF_past_msmts, LSF_steps_forward, datapath,
                LSF_steps_between_msmts=1, LSF_iter=50, LSF_alpha_iter=50, ensembl_size=50):
        
        self.dataobject = le(test_case, variation,
                             GPRP_load='No', 
                             LKFFB_load = 'Yes', LKFFB_path = datapath,
                             AKF_load='No',
                             LSF_load = 'No')
        self.test_case = test_case 
        self.variation = variation  
        self.LSF_past_msmts = LSF_past_msmts 
        self.LSF_steps_forward = LSF_steps_forward  
        self.datapath = datapath 
        self.LSF_steps_between_msmts = LSF_steps_between_msmts 
        self.LSF_err_train_iter = LSF_iter
        self.LSF_ensembl_size = ensembl_size


    def loss_lsf(self, try_alpha0, user_msmt_train=0):
        
        '''Loss function for selecting alpha_0 based min total loss. 
        
        user_msmt_train, if defined, will use one data set for all training.
        user_msmt_train, if undefined, will mean that a new noise realisation and msmt data
        is generated for each new choice of try_alpha_initial i.e. loss_lsf == Bayes Risk'''
        
        if np.sum(user_msmt_train) !=0:
            measurements_train = user_msmt_train
        elif np.sum(user_msmt_train) ==0 :
            measurements_train = get_data(self.dataobject)[0]
        
        training_data = sp.build_training_dataset(measurements_train, past_msmts=self.LSF_past_msmts,
                                                  steps_forward=self.LSF_steps_forward,
                                                  steps_between_msmts=self.LSF_steps_between_msmts)

        weights, errorTrain = sp.gradient_descent(training_data, self.LSF_err_train_iter, alpha_coeff=try_alpha0)
        
        lossval = errorTrain[-1] # based on the last value of err train (not gradient estimate of error train)

        if np.isfinite(lossval):
            return lossval, errorTrain, weights[:, 0]
        print("Lossval not finite")
        return None, errorTrain, weights[:, 0]


    def train_alpha0_manual(self, arr_alphas):

        '''Returns lowest loss alpha_0 from arr_alphas'''

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
        Returns LSF predictions using distribution in both training and validation datasets
        '''

        n_train = self.dataobject.Expt.n_train
        n_start_at = n_train - self.LSF_past_msmts + 1

        #measurements_train, pick_train = get_data(self.dataobject) # implementation in DRAFT 1, DATA_v0/testingLSF/

        macro_weights = []
        macro_predictions = []
        macro_actuals = []
        macro_errorTrain_fore = [] # changes by steps forwards, not a risk avergae.
        macro_truths = []
        macro_data = []
        
        for idx_en in xrange(self.LSF_ensembl_size):

                measurements_train, pick_train = get_data(self.dataobject) # desired implementation in DATA v0
                measurements_val, pick_val = get_data(self.dataobject)
                shape = self.dataobject.LKFFB_macro_truth.shape
                noisetrace_val = self.dataobject.LKFFB_macro_truth.reshape(shape[0]*shape[1], shape[2])[pick_val, :]      
                
                output = doLSF_forecast(measurements_train, measurements_val, pick_alpha0, n_start_at, 
                                        self.LSF_steps_forward, self.LSF_past_msmts, 
                                        steps_between_msmts=self.LSF_steps_between_msmts, num_of_iterGD=num_of_iterGD)
                
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

        '''Returns data matrices for iter runs of LSF using pick_alpha0
        '''

        macro_err_trains = np.zeros((iter_, self.LSF_err_train_iter))
        macro_weights = np.zeros((iter_, self.LSF_past_msmts))

        for idx_run in xrange(iter_):
            macro_err_trains[idx_run, :], macro_weights[idx_run, :] = self.loss_lsf(pick_alpha0)[1:]

        return macro_err_trains, macro_weights



    # def train_alpha0_scipy(self, numIters=10000, fatol=float(10), 
    #                  tol=float(10), factr=10.0, method_='Nelder-Mead', 
    #                  alpha0_0=0.000001):
    #     '''
    #     '''
    #     import scipy.optimize as opt

    #     results = opt.minimize(self.loss_lsf, alpha0_0, method=method_,
    #                             tol =tol, options={'maxiter': numIters,
    #                                                 'disp':True} )

    #     return results['x'], results


    # def train_alpha0_graddescent(self, alpha0_0=0.000001, 
    #                              learnrate=float(10**-7), numIters=100):

    #     alpha_lossTrain = np.zeros(numIters)
    #     try_alpha_traj = np.zeros(numIters)
    #     gradient = np.zeros(numIters)

    #     try_alpha_traj[0]=alpha0_0

    #     n = self.dataobject.Expt.number_of_points

    #     for i in range(0, numIters-1, 1):
    #         alpha_lossTrain[i] = self.loss_lsf(try_alpha_traj[i])  # loss
    #         gradient[i] = np.log((alpha_lossTrain[i] - alpha_lossTrain[i-1])) # / (try_alpha_traj[i]-try_alpha_traj[i-1])
    #         learnrate_i = learnrate / (np.sqrt(1+i)) # decreasing stepsize
    #         try_alpha_traj[i+1] = try_alpha_traj[i] - learnrate_i * gradient[i]

    #             # try:
    #             #     try_alpha_traj[i+1] = try_alpha_traj[i] - learnrate_i * gradient
    #             #     if try_alpha_initial < 0 :
    #             #         raise RuntimeError   # update only if try_alpha-initial >0
    #             # except:
    #             #     try_alpha_traj[i+1] = try_alpha_traj[i]
    #             #     print('Negative weights; loss rejected')                
            
    #     return try_alpha_traj, alpha_lossTrain, gradient