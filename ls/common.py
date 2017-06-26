'''
Created on Thu Apr 20 19:20:43 2017

@author: riddhisw

PACKAGE: ls
MODULE: ls.common
PURPOSE: Returns LSF Predictions using VFrey and SMavadia's LS Filter (statePredictions.py)
METHODS: 
doLSF_forecast -- Returns LSF predictions and trained weights given training and validation data

'''
from __future__ import division, print_function, absolute_import
import numpy as np

def doLSF_forecast(measurements_train, measurements_val, pick_alpha, 
                   n_start_at, n_predict, past_msmts, 
                   steps_between_msmts=1, num_of_iterGD=50):
        '''Returns LSF prediction for n in [n_train, n_train + n_predict] by executing LSF for n_predict different models
        row_at_n_train -- LSF past msmts record, when multipled by weights for the n-th model, gives the n_th step ahead prediction [DIMS: pastmsmts x 1, dtype=float64]
        weights_list -- Sequences of q number of learned AR(q) weights (q==past_msmts); repeated for n-th step ahead prediction models [DIMS: n_predict x pastmsmts x 1, dtype=float64]
        n_step_ahead_prediction -- Net LSF predictions for [n_train, n_train + n_predict] [DIMS: n_predict x 1, dtype=float64]
        n_step_ahead_actual -- Actual noisy msmts for [n_train, n_train + n_predict] [DIMS: n_predict x 1, dtype=float64]
        errorTrain_fore -- Error trains in gradient descent for q weights in weight_list; for n_predict number for models [DIMS: n_predict x num_of_iterGD, dtype=float64]

        n_start_at -- n_train - q + 1 # Time step prior to n_train at which weights are applied to get the n-the step ahead prediction
        '''
        import ls.statePredictions as sp

        weights_list = []
        step_list=[]
        row_at_n_train=[]
        n_step_ahead_prediction=[]
        n_step_ahead_actual=[]

        errorTrain_fore = np.zeros((n_predict, num_of_iterGD))

        for idx_steps in range(0, n_predict, 1): # MODEL CHANGES HERE AS WE CYCLE THROUGH STEPS FWD

                training_data = sp.build_training_dataset(measurements_train, 
                        past_msmts=past_msmts,
                        steps_forward=idx_steps, # training data for n-step ahead
                        steps_between_msmts=steps_between_msmts)

                weights_tuned, errorTrain_fore[idx_steps, :] = sp.gradient_descent(training_data, num_of_iterGD, alpha_coeff=pick_alpha)

                validation_data = sp.build_training_dataset(measurements_val, 
                                                        past_msmts=past_msmts,
                                                        steps_forward=idx_steps, # testing data for n-step ahead
                                                        steps_between_msmts=steps_between_msmts)
                
                if not np.all(np.isfinite(weights_tuned)):
                        print("invalid weights")
                        raise RuntimeError
                
                past_measurements = validation_data[:,1:]
                actual_values = validation_data[:,0]
                predictions = sp.get_predictions(weights_tuned, past_measurements) # predictions for n-step ahead
                
                weights_list.append(weights_tuned)
                row_at_n_train.append(validation_data[n_start_at, :]) # This row in validation data is multipled by weights and gives n-step ahead prediction from t = n_start_at
                n_step_ahead_prediction.append(predictions[n_start_at]) # n-step ahead prediction
                n_step_ahead_actual.append(actual_values[n_start_at])

                # Alternatively, once you have the trained weights for all n_step ahead models:
                # you could try to use [n_train - q, n_train] pts in any truth 
                # to generate predictions. Namely:

                # predictions = np.dot(weights_list, sequence_of_q_past_msmts[::-1]) ## msmts reordered from current to past
                
        return row_at_n_train,  weights_list, n_step_ahead_prediction, n_step_ahead_actual, errorTrain_fore


        