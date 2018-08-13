#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 19:20:43 2017
@author: riddhisw

.. module:: analysis_tools.common

    :synopsis: List of functions required by multiple modules in analysis_tools.

    Module Level Functions:
    ----------------------
        sqr_err : Return the squared error sequence between two input sequences.
        truncate_losses_ : Return indices for all values in a sequence below a threshold.
        get_tuned_params_ : Return lowest Bayes risk params for state estimation and forecasting.

.. moduleauthor:: Riddhi Gupta <riddhi.sw@gmail.com>

"""

from __future__ import division, print_function, absolute_import
import numpy as np


def sqr_err(predictions, truth):
    ''' Return the squared error sequence between two input sequences of same dims.

    Parameters:
    ----------
        predictions (`float64`):  Input sequence of algorithm predictions.
        truth (`float64`):  Input sequence of engineered true values.

    Returns:
    -------
        Distance-squared between elements of input sequences.

    '''
    return (predictions.real - truth.real)**2


def truncate_losses_(list_of_loss_vals, truncation):
    '''
    Return indices for all values in a sequence below a threshold.

    [Helper function for Bayes Risk mapping].

    Parameters:
    ----------
        list_of_loss_vals (`float64`):  Input sequence of values.
        truncation (`int`):  Pre-determined threshold for number of lowest input values to return.

    Returns:
    -------
        indices (`int`):  Indicies of lowest input values.
        losses (`float64`):  Lowest input values.
    '''

    loss_index_list = list(enumerate(list_of_loss_vals))
    low_loss = sorted(loss_index_list, key=lambda x: x[1])
    indices = [x[0] for x in low_loss]
    losses = [x[1] for x in low_loss]

    return indices[0:truncation], losses[0:truncation]


def get_tuned_params_(max_forecast_loss, num_randparams,
                      macro_prediction_errors, macro_forecastng_errors,
                      random_hyperparams_list, truncation):
    ''' Return lowest Bayes risk params for state estimation and forecasting,
        over timesteps and ensemble runs, for each random sample of (sigma, R).

    Parameters:
    ----------
        max_forecast_loss (`int`):  Number of time-steps defining Bayes forecasting loss.
        num_randparams (`int`):  Number of random samples of (sigma, R) pairs.
        macro_prediction_errors (`float`) : Set of state estimation errors.
            [Dim: num_randparams [runs x time_steps]].
        macro_forecastng_errors (`float`) : Set of state estimation errors.
            [Dim: num_randparams x runs x time_steps].
        random_hyperparams_list (`float`): List of random samples of (sigma, R).
        truncation (`int`):  Pre-determined threshold for number of lowest input values to return.

    Returns:
    -------
        means_lists_ : List of states estimation and forecasting Bayes Risk
            values for (sigma, R) samples.
        lowest_pred_BR_pair (`float64`):  Lowest Bayes Risk (sigma, R) pair for state estimation.
        lowest_fore_BR_pair (`float64`):  Lowest Bayes Risk (sigma, R) pair for forecasting.
    '''

    prediction_errors_stats = np.zeros((num_randparams, 2))
    forecastng_errors_stats = np.zeros((num_randparams, 2))

    j = 0
    for j in xrange(num_randparams):

        prediction_errors_stats[j, 0] = np.mean(macro_prediction_errors[j])
        prediction_errors_stats[j, 1] = np.var(macro_prediction_errors[j])
        forecastng_errors_stats[j, 0] = np.mean(macro_forecastng_errors[j, :, 0:max_forecast_loss])
        forecastng_errors_stats[j, 1] = np.var(macro_forecastng_errors[j, :, 0:max_forecast_loss])

    means_list = prediction_errors_stats[:, 0]
    means_list2 = forecastng_errors_stats[:, 0]
    means_lists_ = [means_list, means_list2]

    x_data, y_data = truncate_losses_(means_list, truncation)
    x2_data, y2_data = truncate_losses_(means_list2, truncation)

    lowest_pred_BR_pair = random_hyperparams_list[x_data[0], :]
    lowest_fore_BR_pair = random_hyperparams_list[x2_data[0], :]

    return means_lists_, lowest_pred_BR_pair, lowest_fore_BR_pair
