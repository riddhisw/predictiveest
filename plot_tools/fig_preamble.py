################################################
# Import preamble for all plot_fig*.py
################################################

import sys, numpy as np
import matplotlib

# Packages 
import PyPDF2
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.ticker import MaxNLocator

# Plotting help functions and style sheets
from plot_tools.plot_helper_funcs import plot_normed_means as pnm
from plot_tools.plot_helper_funcs import plot_single_predictions as psp
from plot_tools.plot_helper_funcs import cm2inch, plot_risk_map, plot_risk_trajectories, get_Kalman_LSF_difference, shiftedColorMap, plot_risk_map_2, set_font_sizes

from plot_tools.plot_figstyle_sheet import ALGOLIST, ALGOKEYS, fsize, Fsize, my_dpi, PRED_H, PRED_L, RISK_H, RISK_L, PRED_H2, PRED_L2, COLOURDICT, STYLEDICT, prediczero_lw
from plot_tools.plot_figstyle_sheet import PRED_H_ILL, PRED_L_ILL, RISK_H_ILL, RISK_L_ILL, RISK_H2, RISK_L2, SPEC_EST_H, SPEC_EST_L, ONESHOT_L, ONESHOT_H

# Set global parameters
matplotlib.rcParams['font.size'] = fsize # global
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'
matplotlib.rcParams['axes.linewidth'] = 0.5
matplotlib.rcParams['mathtext.default'] ='regular' # makes mathtext mode Arial. note mathtext is used as ticklabel font in log plots

# Set global tick mark parameters
matplotlib.rcParams['xtick.major.width'] = 0.5
matplotlib.rcParams['ytick.major.width'] = 0.5
matplotlib.rcParams['xtick.labelsize']= fsize
matplotlib.rcParams['ytick.labelsize'] = fsize
matplotlib.rcParams['xtick.minor.visible'] = False
matplotlib.rcParams['ytick.minor.visible'] = False


#import svgutils.transform as sg

# Make fonts arial
#fontProperties = {'family':'sans-serif','sans-serif':['Arial']}
#rc('font',**fontProperties)


#    'weight' : 'normal', 'size' : 8}
# ticks_font = font_manager.FontProperties(family='Helvetica', style='normal',
#    size=sizeOfFont, weight='normal', stretch='normal')
# rc('text', usetex=True)


