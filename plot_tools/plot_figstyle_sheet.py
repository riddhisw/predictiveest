# Plot_figstyle_sheet.py creates consistent figure plotting 
# for all functions defined in plot_helper_funcs.py

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec

##################################################################
# DATA TYPES
##################################################################
ALGOKEYS = ['LSF', 'AKF', 'LKFFB', 'GPRP', 'TRUTH', 'DATA']
ALGOLIST = ['LKFFB', 'AKF']  # for loss maps only


##################################################################
# FONTS AND SIZES
##################################################################
fsize = 8
my_dpi = 400


##################################################################
# DATA COLOURS
##################################################################
# Optimal (sigma, R)
optimal_star='m'

# True band edge
true_bandedg_clr = 'r'

# Computational band edge
lkffb_bandedg_clr = 'teal'

COLOURDICT = {'LSF'  : 'g',
             'AKF'  : 'k',
             'LKFFB' : 'purple',
             'GPRP' : 'sienna',
             'TRUTH' : 'r',
             'DATA' : 'darkblue'}



##################################################################
# CUSTOM LEGEND HANDLES AND LABELS
##################################################################

# Training Region
region_trainng = mpatches.Patch(color='gray', alpha=0.3)

# Random Sigma, R Pairs
pts_hypparams = mlines.Line2D([], [], linestyle='None', color=None, marker='v', markerfacecolor='k', markeredgecolor='k', markersize=7)

# Optimal Sigma, R
pts_optimal = mlines.Line2D([], [], linestyle='None',  color=None, marker='*', markerfacecolor=optimal_star, markeredgecolor=optimal_star, markersize=4, alpha=1)

# Low State Estimation Losses 
lne_se_loss = mlines.Line2D([], [], linestyle='-',  color='tan', marker='o', markerfacecolor='tan', markeredgecolor='tan', markersize=4)

# Low Forecasting Losses 
lne_f_loss = mlines.Line2D([], [], linestyle='-',  color='c', marker='o', markerfacecolor='c', markeredgecolor='c', markersize=4)

# Loss Trajectory for All Other Choices of (Sigma, R)
un_opt_traj = mlines.Line2D([], [], linestyle='-', color='gray')

# Loss Trajectory for Optimal (Sigma, R)
opt_traj = mlines.Line2D([], [], linestyle='-', color=optimal_star)

# LKFFB Computational Band Edge

lne_lkffb_bandedg = mlines.Line2D([], [], linestyle='-', color= lkffb_bandedg_clr)

# True Band Edge
lne_true = mlines.Line2D([], [], linestyle='-', color=true_bandedg_clr)

# Predict Zero Line
lne_predict0 =  mlines.Line2D([], [], linestyle='-', color='darkblue')
pts_data =  mlines.Line2D([], [], linestyle='None',  color=None, marker='x', markerfacecolor='darkblue', markeredgecolor='darkblue', markersize=4)

# Ill Specified LKFFB Basis
region_ill = mpatches.Patch(color='linen', alpha=1.0)

ALPHA_AMPS = 0.5
MSIZE_AMPS = 3

PRED_H = (lne_true, pts_data, lne_predict0)
for algo_type in ALGOKEYS[0:4]:
    vars()['lne_'+algo_type] = mpatches.Patch(color=COLOURDICT[algo_type], alpha=0.7)
    PRED_H += (vars()['lne_'+algo_type], )

PRED_L = ['Truth', 'Data', 'Predict $\mu_{f_n}=0$'] + ALGOKEYS[0:4] 
RISK_H = (pts_hypparams,   lne_se_loss, lne_f_loss, pts_optimal, opt_traj, region_trainng, un_opt_traj)
RISK_L = [r'Random Init. ($\sigma, R$)', 'Low State Est. Risk', 'Low Forecast Risk', r'Tuned ($\sigma, R$)',  'Optimal Traj.', 'Training', 'Unoptimal Traj.']                              

PRED_L_ILL = PRED_L + ['Bad LKFFB Basis']
PRED_H_ILL = PRED_H + (region_ill,)

RISK_L_ILL = RISK_L + ['Bad LKFFB Basis']
RISK_H_ILL = RISK_H + (region_ill,)
