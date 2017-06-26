"""
Created on Thu Apr 20 19:20:43 2017

@author: riddhisw

PACKAGE: data_tools
MODULE: data_tools.load_raw_cluster_data

The purpose of data_tools is to load data and analyse data generated 
by any algorithm (LKFFB, AKF, LSF, GPy) for any scenario (test_case, variation)

MODULE PURPOSE: Class data container that pulls together data output
from different algorithms under one single instance, indexed by test case
and variation number.

"""



from __future__ import division, print_function, absolute_import
import numpy as np
from analysis_tools.experiment import Experiment
from analysis_tools.truth import Truth

ALGO = ['LSF', 'AKF', 'GPRP', 'LKFFB']
FILENAME_DICT = {'AKF': '_BR_AKF_MAP_correctQ_.npz',
                 'LSF': '_LS_Ensemble.npz',
                 'LKFFB': 'BR_Map.npz', 
                 'GPRP': '_GPR_PER_.npz'}


class LoadExperiment(object):
    '''
    LoadExperiment instantiates a data container that takes output from LSF, AKF, LKFFB and 
    GPR npz files and stores all data as class attributes. Each experiment is uniquely indexed by:

    test_case -- scalar. Parameters for test cases are in KalmanParameterRegimes.ods
    variation -- scalar. Parameters for variations in a test case are in KalmanParameterRegimes.ods

    Additionally, if LKFFB is loaded, then LoadExperiment instances Experiment and Truth class objects
    from analysis_tools package. The data attributes of these class objects 
    can then be accessed quickly for analysis and figure plotting. 

    '''

    def __init__(self, test_case, variation, 
                 skip = 1,
                 GPRP_load='Yes', GPRP_path = './LS_Ensemble_Folder/',
                 LKFFB_load = 'Yes', LKFFB_path = './',
                 AKF_load='Yes', AKF_path = './LS_Ensemble_Folder/',
                 LSF_load = 'Yes', LSF_path = './LS_Ensemble_Folder/'):

        self.test_case = test_case
        self.variation = variation

        for item in ALGO:
            setattr(LoadExperiment, item+'_load', vars()[item+'_load'])
            setattr(LoadExperiment, item+'_path', vars()[item+'_path'])
            
            #if item == 'LKFFB':
            #    setattr(LoadExperiment, item+'_path', vars()[item+'_path']+'test_case_'+str(self.test_case)+'/')

            if getattr(LoadExperiment, item+'_load') == 'Yes':

                print(item +': Data Loaded? ' + getattr(LoadExperiment, item+'_load'))
                filename = getattr(LoadExperiment, item+'_path')+'test_case_'+str(self.test_case)+'_var_'+str(self.variation)+FILENAME_DICT[item]
                data_object = np.load(filename) 
                for idx_data in data_object.files:
                    setattr(LoadExperiment, item+'_'+idx_data, data_object[idx_data])

                if item == 'LKFFB':
                    setattr(LoadExperiment, 'Expt', Experiment(getattr(LoadExperiment, 'LKFFB_expt_params')))
                    true_noise_params = [0.0] + [idx_tr for idx_tr in data_object['true_noise_params']]
                    setattr(LoadExperiment, 'Truth', Truth(true_noise_params, num=self.Expt.number_of_points, DeltaT=self.Expt.Delta_T_Sampling))
                    setattr(LoadExperiment, 'MsmtSTD', self.LKFFB_msmt_noise_variance)