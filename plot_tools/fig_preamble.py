################################################
# Import preamble for all plot_fig*.py
################################################

import sys, numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import svgutils.transform as sg


# Plotting tools
from plot_tools.plot_helper_funcs import plot_normed_means as pnm
from plot_tools.plot_helper_funcs import plot_single_predictions as psp
from plot_tools.plot_helper_funcs import cm2inch, plot_risk_map, plot_risk_trajectories

from plot_tools.plot_figstyle_sheet import ALGOLIST, ALGOKEYS, fsize, my_dpi, PRED_H, PRED_L, RISK_H, RISK_L
from plot_tools.plot_figstyle_sheet import PRED_H_ILL, PRED_L_ILL, RISK_H_ILL, RISK_L_ILL 


# Data Analysis functions
from data_tools.load_raw_cluster_data import LoadExperiment as le
from data_tools.data_risk_analysis import build_risk_dict
