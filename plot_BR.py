from __future__ import division, print_function, absolute_import
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import time as t
import os

import sys
sys.path.append("/home/riddhisw/Documents/2017/Mar2017/Scripts") #needs to change for cluster

means_lists_labels = ['Prediction', 'Forecasting']
color_list = ['r', 'g']
test_circ = mlines.Line2D([], [], linestyle='None', color=None, marker='o', markerfacecolor='k', markeredgecolor='k', markersize=10)
pred_circ = mlines.Line2D([], [], linestyle='None',  color=None, marker='o', markerfacecolor='r', markeredgecolor='r', markersize=14, alpha=0.3)
fore_circ = mlines.Line2D([], [], linestyle='None',  color=None, marker='o', markerfacecolor='g', markeredgecolor='g', markersize=14, alpha=0.3)
stars = mlines.Line2D([], [], linestyle='None', color=None, marker='*', markerfacecolor='b', markeredgecolor='b', markersize=10)
crosses = mlines.Line2D([], [],linestyle='None',  color=None, marker='X', markerfacecolor='b',markeredgecolor='b', markersize=10)
pred_lowest = mlines.Line2D([], [], linestyle='None',  color=None, marker='x', markerfacecolor='m', markeredgecolor='m', markersize=14, alpha=0.5)
labels_traj = ['Random Init. Pair', 'Low Prediction Loss', 'Low Forecasting Loss', 'Lowest Prediction Loss', 'CA Optimum', 'COBLA Optimum']


class Plot_BR_Results(object):


    def __init__(self, OneData, figname=None, truncation=20, pre_n_train=2000, pre_max_it=50):
    
        self.OneData = OneData
        self.pre_n_train = pre_n_train
        self.pre_max_it = pre_max_it
        self.figname = figname
        self.truncation = truncation

        self.random_hyperparams_list = None
        self.macro_prediction_errors = None
        self.macro_forecastng_errors = None
        self.macro_truth = None
        self.num_of_datasets = None
        self.n_train = None
        self.n_predict = None
        self.n_testbefore = None
        self.msmt_noise_variance = None
        self.max_it = None
        self.num_of_points = None        
        self.num_randparams = None
        
        self.lowest_pred_BR_pair = None
        self.lowest_fore_BR_pair = None
        
        self.Object = None
        pass
        
        
    def load_data(self):
            
        #print "Data source:", self.OneData
        self.Object = np.load(self.OneData)
        
        self.random_hyperparams_list = self.Object['random_hyperparams_list']
        self.macro_prediction_errors = self.Object['macro_prediction_errors']
        self.macro_forecastng_errors = self.Object['macro_forecastng_errors']
        self.macro_truth = self.Object['macro_truth']
        
        try:
            self.n_train = self.Object['expt_params'][0]
            self.n_predict = self.Object['expt_params'][1]
            self.n_testbefore = self.Object['expt_params'][2]
            self.msmt_noise_variance = self.Object['msmt_noise_variance']
            self.max_it = self.Object['max_it_BR']
        
        except:
            self.n_train = self.pre_n_train
            self.n_testbefore = self.Object['n_testbefore']
            self.n_predict = self.Object['n_predict']
            self.msmt_noise_variance = self.Object['msmt_noise_variance']
            self.max_it = self.pre_max_it
            pass
        
        self.num_of_points = self.n_train + self.n_predict
        self.num_randparams = self.random_hyperparams_list.shape[0]
        pass


    def make_plot(self, savefig='No', fsize = 12, fsize2 = 14):

        self.load_data()
        self.get_tuned_params()

        means_ind = 0
        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15,5))

        for ax in axes[0:2].flat:
            ax.set_yscale('log')
            vars()['x_data_tmp'+str(means_ind)], vars()['y_data_tmp'+str(means_ind)] = self.truncate_losses(self.means_lists_[means_ind])
            ax.plot(vars()['x_data_tmp'+str(means_ind)], vars()['y_data_tmp'+str(means_ind)], 'o', color=color_list[means_ind], markersize=14, alpha=0.2, label='Low Loss at truncation = %s'%(self.truncation))
            ax.plot(xrange(len(self.means_lists_[means_ind])), np.array(self.means_lists_[means_ind]), 'ko')
            ax.set_xlabel('Index Value of Random Hyperparameter Pair')
            ax.set_ylabel('Risk Value (50 Trials)')
            ax.set_title(str(means_lists_labels[means_ind])+' Bayes Risk')


            ax.set_ylim([10**-10,10**7])
            
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                        item.set_fontsize(fsize)
            
            print(  means_lists_labels[means_ind])
            print( "Loss Minimising Point Index:", vars()['x_data_tmp'+str(means_ind)][0])
            print("Optimal Hyper Parameters: ", self.random_hyperparams_list[vars()['x_data_tmp'+str(means_ind)][0], :])
            print( "Loss Value", vars()['y_data_tmp'+str(means_ind)][0])
            means_ind +=1
            
        ax = axes[2]
        ax.set_xscale('log')
        ax.set_yscale('log')
        R = [x[1] for x in self.random_hyperparams_list]
        sigma = [x[0] for x in self.random_hyperparams_list]
        for index in vars()['x_data_tmp'+str(0)]:
            ax.plot(sigma[index], R[index], 'ro', markersize=20, alpha=0.6)
        for index in vars()['x_data_tmp'+str(1)]:
            ax.plot(sigma[index], R[index], 'go', markersize=10, alpha=0.6)
        ax.plot(self.lowest_pred_BR_pair[0], self.lowest_pred_BR_pair[1], 'mx',  markersize=20, mew=5, label='Lowest Prediction Loss')
        ax.plot(sigma, R, 'ko', markersize=5, label='Test Points')
        ax.set_xlabel('Sigma')
        ax.set_ylabel('R')
        ax.set_xlim([10**-15,1000])
        ax.set_ylim([10**-15,1000])
        ax.set_title(r'Prediction (Red) & Forecasting (Green) Risk v. ($\sigma, R$)')
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
                    item.set_fontsize(fsize)
                  
        fig.legend(handles=(test_circ, pred_circ, fore_circ, pred_lowest, stars, crosses), 
                   labels=labels_traj,  bbox_to_anchor=(0.5,0.93), loc=9, ncol=6, frameon=False)
        fig.suptitle('Plot of Low Bayes Risk Region for Measurement Noise Level = %s, Truncation = %s' %(self.msmt_noise_variance, self.truncation), weight='bold')
        fig.subplots_adjust(left=0.05, right=0.95, wspace=0.25, hspace=0.25, top=0.8)
        
        plt.show()
        if self.figname == None:
            self.figname=str(self.OneData)

        if savefig=='Yes':
            fig.savefig(self.figname+'_Trunc_'+str(self.truncation), format="svg")
        #plt.show()
        pass


    def extract_trajectory(self, opt_object_, i_run, j_listnum): #this needs to be moved to an optimisation module
        '''
        Returns a trajectory (nx2 vector) or a final end-point (1x2 point) of an optimisation routine.
        [Helper function for plotting optimization trajectories on 2D Bayes Risk map]
        '''
        
        if j_listnum == None and i_run==0:
            # This extracts a trajectory (x,y) from a list
            
            sigma_ = [x_[0] for x_ in opt_object_]
            R_ = [x_[1] for x_ in opt_object_]
            return sigma_, R_
        
        elif j_listnum != None and j_listnum != 'x':
            # This extracts a trajectory (x,y) from a sequence of pairs of (x,y)_n in CA. j_listnum==3 for opt_var
            sigma_ = [x_[0] for x_ in opt_object_[i_run][j_listnum]]
            R_ = [x_[1] for x_ in opt_object_[i_run][j_listnum]]
            return sigma_, R_
        
        elif j_listnum =='x':
            # This extracts the final (x,y) solution from a set of scipy.optimize.minimize objects
            sigma_ = opt_object_[i_run]['x'][0]
            R_ = opt_object_[i_run]['x'][1]
            return sigma_, R_
        
        print("Nothing returned")
        return None


    def truncate_losses(self, list_of_loss_vals):
        '''
        Returns truncation number of hyperparameters for lowest risk from a sequence of outcomes.
        [Helper function for Bayes Risk mapping]
        '''
        
        loss_index_list = list(enumerate(list_of_loss_vals))
        low_loss = sorted(loss_index_list, key=lambda x: x[1])
        indices = [x[0] for x in low_loss]
        losses = [x[1] for x in low_loss]
        return indices[0:self.truncation], losses[0:self.truncation]

        
    def get_tuned_params(self):
        
        prediction_errors_stats = np.zeros((self.num_randparams, 2)) 
        forecastng_errors_stats = np.zeros((self.num_randparams, 2)) 
        
        j=0
        for j in xrange(self.num_randparams):
            
            prediction_errors_stats[ j, 0] = np.mean(self.macro_prediction_errors[j])
            prediction_errors_stats[ j, 1] = np.var(self.macro_prediction_errors[j])
            forecastng_errors_stats[ j, 0] = np.mean(self.macro_forecastng_errors[j])
            forecastng_errors_stats[ j, 1] = np.var(self.macro_forecastng_errors[j])     
        
        means_list =  prediction_errors_stats[:,0] 
        means_list2 = forecastng_errors_stats[:,0]
        self.means_lists_= [means_list, means_list2]

        x_data, y_data = self.truncate_losses(means_list)
        x2_data, y2_data = self.truncate_losses(means_list2)

        self.lowest_pred_BR_pair = self.random_hyperparams_list[x_data[0], :]
        self.lowest_fore_BR_pair = self.random_hyperparams_list[x2_data[0], :]
        
        print("Optimal params", self.lowest_pred_BR_pair, self.lowest_fore_BR_pair)
        pass

