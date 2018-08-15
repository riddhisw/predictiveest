'''
Created on Thu Apr 20 19:20:43 2017
@author: riddhisw

.. module:: ls.common

    :synopsis: Returns LSF Predictions using VFrey and SMavadia's LS Filter
    (statePredictions.py)

    Module Level Functions:
    ----------------------
        doLSF_forecast :  Return LSF predictions and trained weights given
                training and validation data.

.. moduleauthor:: Riddhi Gupta <riddhi.sw@gmail.com>
'''
from __future__ import division, print_function, absolute_import
import numpy as np

def doLSF_forecast(measurements_train, measurements_val, pick_alpha,
                   n_start_at, n_predict, past_msmts,
                   steps_between_msmts=1, num_of_iterGD=50):
    ''' Return LSF predictions and trained weights given training and validation data.

    doLSF_forecast returns LSF predictions for n in [n_train, n_train + n_predict]
    by executing LSF for n_predict different models.

    Parameters:
    ----------
        measurements_train ('float64`) : Measurement data for training.
        measurements_val ('float64`) : Measurement data for validation.
        pick_alpha ('float64`) : Gradient descent hyper-parameter.
        n_start_at (`int`): n_train - q + 1 # Time step prior to n_train at which
                weights are applied to get the n-the step ahead prediction.
        n_predict (`int`) : Number of time-steps in the forecasting period.
        past_msmts (`int`) : Number of past measurement regressors to include in LSF
                model.
        steps_between_msmts : Number time-steps between measurements.
                Defaults to 1.
        num_of_iterGD : Number of iterations of gradient descent in LSF.
                Defaults to 50.
    Returns:
    -------
        row_at_n_train (`float64`): LSF past msmts record, when multipled by weights
                for the n-th model, gives the n_th step ahead prediction [dims: pastmsmts x 1].
        weights_list (`float64`): Sequences of q number of learned AR(q) weights,
                (q==past_msmts); repeated for n-th step ahead prediction models
                [dims: n_predict x pastmsmts x 1].
        n_step_ahead_prediction (`int`): Net LSF predictions for
                [n_train, n_train + n_predict] [dims: n_predict x 1].
        n_step_ahead_actual (`int`): Actual noisy msmts for [n_train, n_train + n_predict]
                [dims: n_predict x 1]
        errorTrain_fore (`float64`): Error trains in gradient descent for q weights
                in weight_list; for n_predict number for models [dims: n_predict x num_of_iterGD].
    '''
    import ls.statePredictions as sp
    weights_list = []
    step_list = []
    row_at_n_train = []
    n_step_ahead_prediction = []
    n_step_ahead_actual = []
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

        past_measurements = validation_data[:, 1:]
        actual_values = validation_data[:, 0]

        # Predictions for n-step ahead
        predictions = sp.get_predictions(weights_tuned, past_measurements)

        weights_list.append(weights_tuned)

        # This row in validation data is multipled by weights and gives n-step ahead prediction from t = n_start_at
        row_at_n_train.append(validation_data[n_start_at, :])
        n_step_ahead_prediction.append(predictions[n_start_at]) # n-step ahead prediction
        n_step_ahead_actual.append(actual_values[n_start_at])
        # Alternatively, once you have the trained weights for all n_step ahead models:
        # you could try to use [n_train - q, n_train] pts in any truth
        # to generate predictions. Namely:
        # predictions = np.dot(weights_list, sequence_of_q_past_msmts[::-1]) ## msmts reordered from current to past

    return row_at_n_train,  weights_list, n_step_ahead_prediction, n_step_ahead_actual, errorTrain_fore
    