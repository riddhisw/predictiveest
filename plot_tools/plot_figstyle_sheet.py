# Plot_figstyle_sheet.py creates consistent figure plotting 
# for all functions defined in plot_helper_funcs.py

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

##################################################################
# FONTS AND SIZES
##################################################################
fsize = 8
Fsize = 10 # changed locally using plot_helper_funcs.set_font_sizes()
mpl.rcParams['font.size'] = fsize # global

my_dpi = 400

# Bayes Risk Markers and Lines
algorithm_lw = 1.
prediczero_lw = 1.
algomarkersize= 2.

# Single Prediction Markers
msmt_marker = 2
msmt_marker_s = 'o'
predmarker = 1. # marker size of forepred_s
state_lw = 0.0 # line width of statepred_s
statepred_s = "None"
pred_lw = 1.0
truthline_lw = 1.0

# Spectrum Estimation Amplitude Markers and Lines
ALPHA_AMPS = 0.5
MSIZE_AMPS = 3
ampltiude_s = 'o'
truthline = '-'

# Loss Plot Markers and Lines
datamarker = msmt_marker
pts_hypparams_s = 'o'
optimal_star_s = '*'
lossregion_se_s = 'o'
lossregion_fe_s = 'o'
lossregion_se = 15
lossregion_fe = 10
opt_marker = 10

##################################################################
# DATA TYPES
##################################################################
ALGOKEYS = ['AKF', 'LKFFB', 'LSF', 'GPRP', 'TRUTH', 'DATA']
ALGOLIST = ['LKFFB', 'AKF']  # for loss maps only


##################################################################
# DATA COLOURS
##################################################################

# Set up Color Pallette usign Vega2b and Vega 2c
color_pallete = []
color_pallete.append(plt.cm.Vega20b(np.arange(20)))
color_pallete.append(plt.cm.Vega20c(np.arange(20))) 
color_pallete = np.asarray(color_pallete).reshape(40,4)


# Optimal (sigma, R)
optimal_star=color_pallete[24]


# Single predictions and normed mean styles 
COLOURDICT = {'LSF'  : color_pallete[28],
             'AKF'  : color_pallete[14],
             'LKFFB' : color_pallete[2],
             'GPRP' : color_pallete[20],
             'GPRP2' : color_pallete[24],
             'TRUTH' : color_pallete[36],
             'DATA' : color_pallete[37]}
             

LINE = {'LSF'  : ':',
             'AKF'  : '-',
             'LKFFB' : ':',
             'GPRP' : '--'}
             
MARKR = {'LSF'  : '^',
             'AKF'  : '',
             'LKFFB' : 's',
             'GPRP' : 'd'}
             
STYLEDICT = {'LSF'  : MARKR['LSF'] + LINE['LSF'],
             'AKF'  : MARKR['AKF'] + LINE['AKF'],
             'LKFFB': MARKR['LKFFB'] + LINE['LKFFB'],
             'GPRP' : MARKR['GPRP'] + LINE['GPRP']}
             
# Loss Regions
datamarker_c = COLOURDICT['DATA']
lossregion_se_c = color_pallete[23]
lossregion_fe_c = color_pallete[19]

# Computational band edge
lkffb_bandedg_clr = color_pallete[2] #COLOURDICT['LKFFB'] 

# True band edge
true_bandedg_clr = COLOURDICT['TRUTH']

##################################################################
# CUSTOM LEGEND HANDLES AND LABELS
##################################################################

# Training Region
region_trainng = mpatches.Patch(color='gray', alpha=0.3)

# Random Sigma, R Pairs
pts_hypparams = mlines.Line2D([], [], linestyle='None', color=None, marker=pts_hypparams_s, markerfacecolor=COLOURDICT['DATA'], markeredgecolor=COLOURDICT['DATA'], markersize=datamarker)

# Optimal Sigma, R
pts_optimal = mlines.Line2D([], [], linestyle='None',  color=None, marker=optimal_star_s, markerfacecolor=optimal_star, markeredgecolor=optimal_star, markersize=opt_marker, alpha=1)

# Low State Estimation Losses
lne_se_loss = mlines.Line2D([], [], linestyle='None',  color=lossregion_se_c, marker=lossregion_se_s, markerfacecolor=lossregion_se_c, markeredgecolor=lossregion_se_c, markersize=lossregion_se)

# Low Forecasting Losses 
lne_f_loss = mlines.Line2D([], [], linestyle='None',  color=lossregion_fe_c, marker=lossregion_fe_s, markerfacecolor=lossregion_fe_c, markeredgecolor=lossregion_fe_c, markersize=lossregion_fe)

# Loss Trajectory for All Other Choices of (Sigma, R)
un_opt_traj = mlines.Line2D([], [], linestyle='-', color='gray')

# Loss Trajectory for Optimal (Sigma, R)
opt_traj = mlines.Line2D([], [], linestyle='-', color=optimal_star)

# LKFFB Computational Band Edge
lne_lkffb_bandedg = mlines.Line2D([], [], linestyle=':', color= lkffb_bandedg_clr)

# True Band Edge
lne_true = mlines.Line2D([], [], linestyle=':', color=true_bandedg_clr)

# Predict Zero Line 
lne_predict0 =  mlines.Line2D([], [], linestyle='-', color=COLOURDICT['DATA'])

# Msmt Data 
pts_data =  mlines.Line2D([], [], linestyle='None',  color=None, marker=msmt_marker_s, markerfacecolor=COLOURDICT['DATA'], markeredgecolor=COLOURDICT['DATA'], markersize=msmt_marker, alpha=0.5)

# Ill Specified LKFFB Basis
region_ill = mpatches.Patch(color='linen', alpha=1.0)

# Single Shots
singleshots =  mlines.Line2D([], [], linestyle='None',  color=None, marker='d', markerfacecolor=COLOURDICT['DATA'], markeredgecolor=COLOURDICT['DATA'], markersize=msmt_marker, alpha=0.8)

# Decimated Shots
decimateshots =  mlines.Line2D([], [], linestyle='None',  color=None, marker=msmt_marker_s, markerfacecolor=color_pallete[0], markeredgecolor=color_pallete[0], markersize=msmt_marker, alpha=0.8)

# P(beta, tau) theory
p_betatau_lne = mlines.Line2D([], [], linestyle='--', lw= truthline_lw, c=color_pallete[13])

# LEGEND LABELS AND HANDLES -- LISTS --- SPECTRUM ESTIMATION

SPEC_EST_H = (lne_true, lne_lkffb_bandedg)
SPEC_EST_L =[r'True Band Edge $Jf_0$', r'$f_{MAX} \equiv f_s/r_{Nqyuist}, f_s =1/\Delta t$']

# LEGEND LABELS AND HANDLES -- LISTS --- SINGLE PREDICTION - ALL ALGORITHMS
PRED_H = (lne_true, pts_data, lne_predict0)
for algo_type in ALGOKEYS[0:4]:
    vars()['lne_'+algo_type] = mlines.Line2D([], [], linestyle=LINE[algo_type], marker=MARKR[algo_type], color=COLOURDICT[algo_type], markerfacecolor=np.asarray(list(COLOURDICT[algo_type][0:3]) + [0.6]))
    PRED_H += (vars()['lne_'+algo_type], )

PRED_L = ['Truth', 'Msmts', 'Predict $\mu_{f_n}=0$'] + ALGOKEYS[0:4] 

# LEGEND LABELS AND HANDLES -- LISTS --- SINGLE PREDICTION - ALGORITHMS EXCLUDING GPRP

PRED_H2 = (lne_true, pts_data,)
for algo_type in ALGOKEYS[0:3]:
    vars()['lne_'+algo_type] = mlines.Line2D([], [], linestyle=LINE[algo_type], marker=MARKR[algo_type], markersize=predmarker,
                                             color=COLOURDICT[algo_type], markerfacecolor=np.asarray(list(COLOURDICT[algo_type][0:3]) + [0.6]))
    PRED_H2 += (vars()['lne_'+algo_type], )
    
#PRED_H2 += (lne_predict0,)
PRED_L2 = ['Truth', 'Msmts'] + ALGOKEYS[0:3] #+ ['Predict $\mu_{f_n}=0$']

# LEGEND LABELS AND HANDLES -- LISTS --- LOSS MAPS AND RISK TRAJECTORIES - AKF, LKFFB

RISK_H = (pts_hypparams,   lne_se_loss, lne_f_loss, pts_optimal, opt_traj, region_trainng, un_opt_traj)
RISK_L = [r'Random ($\sigma, R$)', 'Low State Est. Risk', 'Low Forecast Risk', r'Tuned ($\sigma, R$)',  'Optimal Traj.', 'Training', 'Unoptimal Traj.']

# LEGEND LABELS AND HANDLES -- LISTS --- LOSS MAPS ONLY - AKF, LKFFB
RISK_H2 = (pts_hypparams,   lne_se_loss, lne_f_loss, pts_optimal)
RISK_L2 = [r'Random ($\sigma, R$)', 'Low State Est. Risk', 'Low Prediction Risk', r'Tuned ($\sigma^*, R^*$)']                              
                             
# LEGEND LABELS AND HANDLES -- LISTS --- MISC

ONESHOT_L = [r'$P(\beta, \tau)$', 'Msmts', 'F. Msmts']
ONESHOT_H = (p_betatau_lne, singleshots, decimateshots,)

PRED_L_ILL = PRED_L + ['Bad LKFFB Basis']
PRED_H_ILL = PRED_H + (region_ill,)

RISK_L_ILL = RISK_L + ['Bad LKFFB Basis']
RISK_H_ILL = RISK_H + (region_ill,)
