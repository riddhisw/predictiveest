from __future__ import division, print_function, absolute_import

import matplotlib.pyplot as plt
import numpy as np

from ML_Experiment import Experiment

basis_list = ['A', 'B', 'C'] 
prediction_method_list =["ZeroGain", "PropForward"]
count_steps_DICT = {'A': 0, 'B': 2, 'C': 4}
kf_colour_list = ['pink', 'r', 'b','c', 'lightgreen', 'g']

class Plot_KF_Results(Experiment):


    def __init__(self, exp_params, filename_and_path):

        Experiment.__init__(self, exp_params)
        self.filename_and_path = filename_and_path
        self.Normalised_Means_ = None
        self.Normalised_Predict_Zero_Means_ = None
        self.KF_Predictions_Matrix = None
        self.data_loaded_once = False
        self.msmt_noise_variance = None
        self.max_run = None
        self.step_forward_limit_allbasis = []

        pass


    def load_data(self):

        self.KFObject = np.load(self.filename_and_path)
        
        KF_Error_Means_ = self.KFObject['KF_Error_Means']
        Predict_Zero_Means_ = self.KFObject['Predict_Zero_Means']
        self.KF_Predictions_Matrix = self.KFObject['KF_Predictions_Matrix']
        self.truth_datasets = self.KFObject['truth_datasets']
        self.msmt_noise_variance = self.KFObject['msmt_noise_variance']
        
        self.max_run = self.truth_datasets.shape[1]
        
        self.Normalised_Means_ = KF_Error_Means_/ Predict_Zero_Means_
        self.Normalised_Predict_Zero_Means_ = Predict_Zero_Means_ / Predict_Zero_Means_
        self.data_loaded_once = True

        pass


    def show_one_prediction(self, savefig='Yes', fsize=14):

        if self.data_loaded_once == False:
            self.load_data()
        figname=str(self.filename_and_path)
        
        pick_rand_run = int(np.random.uniform(low=0, high=self.max_run))
        truth = self.truth_datasets[self.n_train - self.n_testbefore : self.n_train + self.n_predict, pick_rand_run]
        x_data = self.Time_Axis[self.n_train - self.n_testbefore: self.n_train + self.n_predict ]
        x_data_p = x_data[0 :self.n_testbefore]
        x_data_f = x_data[self.n_testbefore:]
        y_data = self.KF_Predictions_Matrix[0, :, pick_rand_run]
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,8))
        ax.set_title('Single Prediction Run = %s'%(pick_rand_run))
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Signal (Signal Units )')
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(fsize)
        ax.plot(x_data_p, y_data[0: self.n_testbefore], 'bx', label='One Step Ahead Predictions')
        ax.plot(x_data_f, y_data[self.n_testbefore:], 'go', label='n-th Step Ahead Forecasts')
        ax.plot(x_data, truth, 'r', label='Truth')
        ax.axvline(self.n_train*self.Delta_T_Sampling, linestyle='--', color='gray', label='Training Ends')
        ax.legend(loc=1)
        plt.show()

        if savefig=='Yes':
            fig.savefig(figname+'_single_pred_', format="svg")
        pass


    def count_steps(self, figx=None, Basis='A'):
        
        if self.data_loaded_once == False:
            return "please load data and then call again"
        
        counter =0
        basis_index = count_steps_DICT[Basis]
        
        if figx != None:
            for x in figx.Normalised_Means_[basis_index, figx.n_testbefore:]:
                if x > 1:
                    break
                counter +=1
            return counter
        
        for x in self.Normalised_Means_[basis_index, self.n_testbefore:]:
            if x > 1:
                break
            counter +=1
        return counter



    def make_plot(self, savefig='Yes', fsize=14):

        if self.data_loaded_once == False:
            self.load_data()
        
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15,8))
        figname=str(self.filename_and_path)

        choice_counter = 0
        self.step_forward_limit_allbasis = []

        ax.set_title(figname)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Log(E[squared error]) (Log Signal Units^2 )')
        ax.set_yscale('log')
        ax.set_ylim([10**(-5), 10**3])

        for choice1 in basis_list:
            
            step_forward_limit = self.count_steps(Basis=choice1)
            self.step_forward_limit_allbasis.append(step_forward_limit)
            
            for choice2 in prediction_method_list:

                kf_choice_labels = 'KF Basis '+ choice1 + ': '+choice2

                ax.plot(self.Time_Axis[self.n_train - self.n_testbefore : self.n_train + self.n_predict ], self.Normalised_Means_[choice_counter, :], 
                        c = kf_colour_list[choice_counter], label=kf_choice_labels+' (parity @ '+str(step_forward_limit)+ ' stps fwd)')
                choice_counter +=1

        ax.plot(self.Time_Axis[self.n_train - self.n_testbefore : self.n_train + self.n_predict ], self.Normalised_Predict_Zero_Means_, c = 'k', label='Predict Noise Mean')

        ax.axvline(self.n_train*self.Delta_T_Sampling, linestyle='--', color='gray', label='Training Ends')
        ax.legend(loc=2, fontsize=fsize)

        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                        item.set_fontsize(fsize)
        plt.show()
        

        if savefig=='Yes':
            fig.savefig(figname+'_EA_', format="svg")

        #plt.show()


       
