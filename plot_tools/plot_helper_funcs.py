'''
Plot_helper_funcs.py is a list of functions that help produce figures 
reported in analysis. 

'''
import matplotlib.pyplot as plt, numpy as np, matplotlib
import os
from PyPDF2 import PdfFileReader, PdfFileWriter

# from data_tools package
from data_tools.data_risk_analysis import build_risk_dict, riskmapdata, norm_risk, analyse_kalman_errs
from data_tools.data_tuned_run_analysis import TUNED_RUNS_DICT
from data_tools.load_raw_cluster_data import LoadExperiment as le 

# from plot_tools package
from plot_tools.plot_figstyle_sheet import *


##################################################################
# RISK MAP AND TRAJECTORIES
##################################################################


def plot_risk_map_2(figax1, ALGOLIST, test_case, variation, path,
                    figax2=None, 
                    xlim=[-11, 3], ylim = [-11, 3],
                    fstep=50, sstep=50, lowloss=20):
    '''Wrapper function for plot_risk_map to make loss maps for different test cases and types (AKF or LKFFB).'''

    testobj = le(test_case, variation, 
                 AKF_path=path,
                 LKFFB_path=path,
                 LSF_path='', GPRP_path='',
                 GPRP_load='No', LSF_load='No', AKF_load='Yes', LKFFB_load='Yes')
                 
    RISKDICT = build_risk_dict(testobj)
    
    figaxes_ = [figax1, figax2] # max of two algorithms (LKFFB or AKF)
    
    for idx_count in xrange(len(ALGOLIST)):
        algotype = ALGOLIST[idx_count]
        figaxes_[idx_count] = plot_risk_map(figaxes_[idx_count], algotype, RISKDICT, 
                  fstep=fstep, sstep=sstep, lowloss=lowloss, 
                  xlim=xlim, ylim=ylim)
    
    return figaxes_[0], figaxes_[1]
    
    
    
    
def plot_risk_map(figaxes, algotype, RISKDICT, 
                  fstep=50, sstep=50, lowloss=20, 
                  xlim=[-11, 3], ylim = [-11, 3]):
    
    '''Plots risk map on figaxes.
    Algotype = 'AKF' or 'LKFFB'
    RISKDICT = Structured data for a testcase, variation 
    fstep = max forecasting steps forward
    sstep = max state estimation steps forward
    lowloss = the lowest'lowloss' number of loss values are shaded as low loss regions
    '''
    
    p_err, hyp, f_err = RISKDICT[algotype][0:3]
    
    s_sigma, s_R, f_sigma, f_R = riskmapdata(p_err, 
                                             f_err, 
                                             hyp, 
                                             maxforecaststps=fstep, 
                                             maxstateeststps=sstep)[0:4]
    
    figaxes.set_xlim([10**xlim[0],10**xlim[1]])
    figaxes.set_ylim([10**ylim[0],10**ylim[1]])
    figaxes.set_xscale('log')
    figaxes.set_yscale('log')
    
    # As per algorithm code for AKF, LKFFB, we implemented sigma and R^2. To make both units
    # variances, we need to square all the sigmas. 
    
    figaxes.plot(s_sigma[0:lowloss]**2, s_R[0:lowloss], lossregion_se_s, c=lossregion_se_c, markersize=lossregion_se, alpha=0.7)
    figaxes.plot(f_sigma[0:lowloss]**2, f_R[0:lowloss], lossregion_fe_s, c=lossregion_fe_c, markersize=lossregion_fe, alpha=0.7)
    figaxes.plot(s_sigma**2, s_R, pts_hypparams_s, c=COLOURDICT['DATA'], markersize=datamarker, alpha=1.0)
    figaxes.plot(s_sigma[0]**2, s_R[0], optimal_star_s, color=optimal_star, markersize=opt_marker, mew=2)
    figaxes.tick_params(direction='in', which='both')
    
    figaxes.set(title=algotype, xlabel=r'$\sigma^2 \quad [f_n^2] $', ylabel =r'$R \quad [f_n^2]$')
    
    figaxes.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=3))
    figaxes.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=3))
    
    return figaxes


def plot_risk_trajectories(figaxes, algotype, RISKDICT, 
                  fstep=50, sstep=50, lowloss=20, 
                  xlim=None, ylim = [1, 7]): 
    '''Plots risk trajectories on figaxes.
    Algotype = 'AKF' or 'LKFFB'
    RISKDICT = Structured data for a testcase, variation 
    fstep = max forecasting steps forward
    sstep = max state estimation steps forward
    lowloss = the lowest'lowloss' number of loss values are shaded as low loss regions
    '''
    
    p_err, hyp, f_err = RISKDICT[algotype][0:3]
    s_traj, f_traj = riskmapdata(p_err, 
                                 f_err, 
                                 hyp, 
                                 maxforecaststps=fstep, 
                                 maxstateeststps=sstep)[6:8]

    for idx_traj in range(len(s_traj)-1, -1, -1):
        
        if idx_traj < lowloss:
            figaxes.plot(range(-sstep, 0, 1), s_traj[idx_traj][0:fstep], '-', markersize=4, alpha=0.4, c=lossregion_se_c) # this is wrong
            figaxes.plot(range(0, fstep, 1), f_traj[idx_traj][0:fstep], '-', markersize=4, alpha=0.4, c=lossregion_fe_c)
            
        elif idx_traj >= lowloss: 
            figaxes.plot(range(-sstep, 0, 1), s_traj[idx_traj][0:fstep], '-', markersize=4, alpha=0.1, c='k') # this is wrong
            figaxes.plot(range(0, fstep, 1), f_traj[idx_traj][0:fstep], '-', markersize=4, alpha=0.1, c='k')

    figaxes.plot(range(-sstep, 0, 1), s_traj[0], '-', markersize=4, c=optimal_star) 
    figaxes.plot(range(0, fstep, 1), f_traj[0][0:fstep], '-', markersize=4, c=optimal_star)
    
    
    # Reformat x-axis to depict training region
    figaxes.axvspan(-sstep,0, color='gray', alpha=0.3)
    
    figaxes.set_xlim([-sstep - 10, fstep])  # start 10 points before sstep, add n_train label
    xtickslabels =[x.get_text() for x in figaxes.get_xticklabels()]
    xtickslabels[0] = str(r'$-n_{T}$')
    xtickvalues = [int(x) for x in figaxes.get_xticks()]
    xtickslabels[1:] = xtickvalues[1:]
    
    figaxes.set_xticklabels(xtickslabels)
    figaxes.axvspan(-sstep-10, -sstep, color='gray', alpha=0.6)
    
    figaxes.set_yscale('log')
    figaxes.set_ylim([10**ylim[0], 10**ylim[1]])
    figaxes.tick_params(direction='in', which='both')
    figaxes.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=6))
    figaxes.set(title=algotype, xlabel=r'T. Stps [num]', ylabel=r'log($\langle (f_n -\hat{f_n})^2 \rangle_D$) [log($f_n^2$)]')
    
    return figaxes


##################################################################
#   SINGLE PREDICTIONS
##################################################################

def calc_undersampling(testobj):
    
    # This factor scales LKFFB Amplitudes when undersampling is present.
    # The scaling is necessary else the true PSD is much smaller than Kalman Amplitudes
    # The undersampled regimes forces the filter to weight Kalman frequencies much higher 
    # to compensate for the true frequencies it can't see.
    
    # In a practical example, we do not have access to this true constant, and hence
    # we are merely confirming a numerical hypothesis.
    
    true_undersampl_const = 0.0
    true_undersampl_const = testobj.LKFFB_kalman_params[4]/testobj.Truth.f0
    if true_undersampl_const < 1.0:
        true_undersampl_const = 1.0 # This correction doesn't apply to adequately sampled regimes
    #print("UNDSAMPL_FUDGE=", UNDSAMPL_FUDGE)
    return true_undersampl_const


def plot_single_predictions(figaxes, figaxes_amps, ALGOLIST, test_case, variation, path,
                            GPRP_load='Yes', LSF_load='Yes', AKF_load='Yes', LKFFB_load='Yes',
                            fstep=50, sstep=50, lowloss=20, lgd_loc=4, nt_label=10, 
                            ylim_amps = [-4.5, 1], yscale='linear', undersampl_scale_on=0, sig_scale_on=1):
    
    '''Returns single predictions from all algorithms, spectral estimates from AKF/LSF, LKKFB'''

    testobj = le(test_case, variation, 
                 AKF_path=path, AKF_load=AKF_load,
                 LKFFB_path=path, LKFFB_load=LKFFB_load,
                 LSF_path=path, LSF_load=LSF_load,
                 GPRP_path=path, GPRP_load=GPRP_load)
    
    # scaling factor for signals if Truth.alpha != 1.0 [asthetic scaling for plotting]
    signal_scaling = 1.0
    if LKFFB_load == 'Yes' and sig_scale_on != 0 :
        signal_scaling = 1.0 / testobj.Truth.alpha
        
    # make (and scale) data
    pickone = int(np.random.uniform(low=0, high=49))
    truth = testobj.LSF_macro_truths[pickone, :]
    signal = testobj.LSF_macro_data[pickone, :]

    ntb = testobj.Expt.n_testbefore
    ntn = testobj.Expt.n_train
    x_axis = range(-sstep, fstep, 1)
    
    # scaling factor for amplitudes in the undersampling regime [Ref: calc_undersampling for rationale]
    true_undersampl_const = 1.0
    if undersampl_scale_on != 0 :
        true_undersampl_const = calc_undersampling(testobj)
    
    # plot data
    figaxes.plot(x_axis[ : sstep], signal[ntn - sstep : ntn]*signal_scaling, msmt_marker_s, c= datamarker_c, label='Data', markersize=msmt_marker, alpha=0.7)
    
    # store ylim for each algorithm for plotting later on
    ylim_list = []
    
    for algo_type in ALGOLIST:
        
        print(algo_type)
        
        if algo_type == 'LKFFB' or algo_type == 'AKF':

            RISKDICT=build_risk_dict(testobj)
            opt_sigma, opt_R = analyse_kalman_errs(RISKDICT[algo_type][0],
                                                   RISKDICT[algo_type][1],
                                                   50)[2:]

            KWGS = {'opt_sigma': opt_sigma[0], 'opt_R': opt_R[0]}
            faxis, amp, thirdVar, pred_ = TUNED_RUNS_DICT[algo_type](testobj, signal, **KWGS)
            
            pred = pred_*signal_scaling
            ylim_list.append(center_y_axis(pred))
            
            # Plot predictions (Kalman)
            figaxes.plot(x_axis[ : sstep], pred[ntb - sstep : ntb ], statepred_s, lw=state_lw, color=COLOURDICT[algo_type])
            figaxes.plot(x_axis[ sstep : ], pred[ntb : ntb + fstep], STYLEDICT[algo_type],
                         color=COLOURDICT[algo_type], markerfacecolor=np.asarray(list(COLOURDICT[algo_type][0:3]) + [0.6]), 
                         markersize=predmarker, lw=pred_lw,
                         label=algo_type)
            
            # Plot amplitude estimates
            if algo_type == 'LKFFB':
                
                # LKFFB
                figaxes_amps.plot(faxis[0], amp[0]*(1.0/true_undersampl_const), ampltiude_s, 
                                  c=COLOURDICT[algo_type], alpha=ALPHA_AMPS, markersize= MSIZE_AMPS,
                                  label=' T.Pw: %.1e' %(thirdVar[0]))
                # Truth
                figaxes_amps.plot(faxis[1], amp[1], truthline, c=COLOURDICT['TRUTH'], label='T.Pw: %.1e' %(thirdVar[1]))     
                
            else:
                truncatedata = int(faxis.shape[0]*300.0/ faxis[-1])
                figaxes_amps.plot(faxis[0: truncatedata], thirdVar[0: truncatedata], ampltiude_s, 
                                  c=COLOURDICT[algo_type], alpha=ALPHA_AMPS, markersize= MSIZE_AMPS,
                                  label=' T.Pw: %.1e' %(np.sum(thirdVar)))
        else:

            # Plot predictions (Non Kalman)
            
            pred = TUNED_RUNS_DICT[algo_type](testobj, signal)*signal_scaling
            ylim_list.append(center_y_axis(pred))
            
            if algo_type == 'LSF':
                figaxes.plot(x_axis[sstep :], pred[: fstep], STYLEDICT[algo_type], markersize= predmarker, lw=pred_lw,
                             color=COLOURDICT[algo_type], markerfacecolor=np.asarray(list(COLOURDICT[algo_type][0:3]) + [0.6]), 
                             label=algo_type)
            
            elif algo_type == 'GPRP' and testobj.GPRP_load != 'No':
                figaxes.plot(x_axis[ : sstep], pred[ ntb - sstep : ntb ], statepred_s, lw=state_lw, color=COLOURDICT[algo_type])
                figaxes.plot(x_axis[ sstep : ], pred[ntb : ntb + fstep ], STYLEDICT[algo_type],
                             markersize=predmarker, lw=pred_lw,
                             color=COLOURDICT[algo_type], markerfacecolor=np.asarray(list(COLOURDICT[algo_type][0:3]) + [0.6]),
                             label=algo_type)

    # Config x-axis for amplitudes with bandwidth assumptions
    
    bandedge = testobj.Truth.f0*(testobj.Truth.J-1)*2.0*np.pi
    compedge = testobj.Expt.bandwidth*2.0*np.pi
    figaxes_amps.axvline(x=bandedge, ls=':', c=true_bandedg_clr)#, label= 'True Band Edge')
    figaxes_amps.axvline(x=compedge, ls=':', c=lkffb_bandedg_clr)#, label= 'KF Basis Ends')
    
    # Config x axis for predictions
    # figaxes.axvspan(-sstep, 0, color='gray', alpha=0.1)
    figaxes.set_xlim([-sstep - nt_label, fstep])  # start 10 points before sstep, add n_train label
    xtickslabels =[x.get_text() for x in figaxes.get_xticklabels()]
    xtickslabels[0] = str(r'$-n_{T}$')
    xtickvalues = [int(x) for x in figaxes.get_xticks()]
    xtickslabels[1:] = xtickvalues[1:]
    figaxes.set_xticklabels(xtickslabels)
    # figaxes.axvspan(-sstep - nt_label, -sstep, color='gray', alpha=0.2)
    
    # Config y axis for amplitudes
    figaxes_amps.set_yscale('log')
    figaxes_amps.set_ylim([10**ylim_amps[0], 10**ylim_amps[1]])
    figaxes_amps.tick_params(direction='in', which='both')
    
    # Config y axis for predictions
    #figaxes.ticklabel_format(style='sci', scilimits=(0,2), axis='y')
    figaxes.tick_params(direction='in', which='both')
    final_y_lim = np.max(np.asarray(ylim_list)) # don't cut any algorithm out of plotting area
    figaxes.set_ylim([-final_y_lim, final_y_lim]) # symmetric axis about y=0
    
    # Set labels
    figaxes_amps.legend(loc=lgd_loc, fontsize=6)
    figaxes_amps.set(xlabel=r'$\omega$ [rad]', ylabel=r'$S(\omega)$ [$f_n^2$/(rad $s^{-1}$)]')
    figaxes.set(xlabel='Time Steps [num]', ylabel=r'Predictions [$f_n$]')
    
    # plot truth
    figaxes.plot(x_axis, truth[ntn - sstep : ntn + fstep]*signal_scaling, c=COLOURDICT['TRUTH'], label='Truth', lw=truthline_lw)
    
    return figaxes, figaxes_amps


##################################################################
#   NORMED  BAYES RISK - ALL ALGORITHMS
##################################################################

def get_n_train(testobj):
    '''Returns the value of n_train used by a test_case. '''
    try:
        n_train = testobj.Expt.n_train
    except:
        try:
            n_train = testobj.LSF_n_train
        except:
            print("Using hardcoded value for n_train at 2000")
            n_train = 2000
    return n_train
    

def plot_normed_means(figaxes, inset, ALGOLIST, test_case, variation, path,
                      GPRP_load='Yes', LSF_load='Yes', AKF_load='Yes', LKFFB_load='Yes',
                      fstep=50, sstep=50, lowloss=20, 
                      ylim = [-11, 3], yscale='log'):

    testobj = le(test_case, variation, 
                 AKF_path=path, AKF_load=AKF_load,
                 LKFFB_path=path, LKFFB_load=LKFFB_load,
                 LSF_path=path, LSF_load=LSF_load,
                 GPRP_path=path, GPRP_load=GPRP_load)
    RISKDICT = build_risk_dict(testobj)
    
    n_train = get_n_train(testobj)

    print(testobj.test_case, testobj.variation)

    for algo_type in ALGOLIST:
        
        #print(algo_type)
        p_err, hyp, f_err, truth, lsf = RISKDICT[algo_type]

        if lsf != 0: # proceed only if data has been loaded

            opt_idx=0

            if algo_type == 'AKF' or algo_type == 'LKFFB' :

                opt_idx = analyse_kalman_errs(p_err, hyp, sstep)[0][0]

            norm_s, norm_f = norm_risk(f_err[opt_idx, ..., 0:fstep],  # opt_idx is zero for LSF and GPRP
                                       truth[opt_idx, ...], 
                                       n_train,
                                       opt_state_err=p_err[opt_idx, ...],
                                       LSF=lsf)
            if lsf == 'No':
                figaxes.plot(range(-norm_s.shape[0]-1,-1,1), norm_s, STYLEDICT[algo_type], 
                             lw=algorithm_lw, markersize=algomarkersize, markerfacecolor=np.asarray(list(COLOURDICT[algo_type][0:3]) + [0.6]),
                             c=COLOURDICT[algo_type], label=algo_type+' State Est')

            figaxes.plot(range(0,norm_f.shape[0],1), norm_f, STYLEDICT[algo_type], 
                               lw=algorithm_lw, markersize=algomarkersize, markerfacecolor=np.asarray(list(COLOURDICT[algo_type][0:3]) + [0.6]),
                               c=COLOURDICT[algo_type], label=algo_type+' Forecast')

            inset.plot(range(0,norm_f.shape[0],1), norm_f, STYLEDICT[algo_type], 
                             lw=algorithm_lw, markersize=algomarkersize, markerfacecolor=np.asarray(list(COLOURDICT[algo_type][0:3]) + [0.6]),
                             c=COLOURDICT[algo_type])
        

    
    # Config x axis for main graph
    figaxes.axvspan(-sstep,0, color='gray', alpha=0.1)
    figaxes.set_xlim([-sstep - 10, fstep])  # start 10 points before sstep, add n_train label
    xtickslabels =[x.get_text() for x in figaxes.get_xticklabels()]
    xtickslabels[0] = str(r'$-n_{T}$')
    xtickvalues = [int(x) for x in figaxes.get_xticks()]
    xtickslabels[1:] = xtickvalues[1:]
    
    figaxes.set_xticklabels(xtickslabels)
    figaxes.axvspan(-sstep-10, -sstep, color='gray', alpha=0.2)
    
    # Config y axis for main graph
    figaxes.set_yscale(yscale)
    figaxes.set_ylim([10**ylim[0], 10**ylim[1]])
    figaxes.tick_params(direction='in', which='both')

    # Config inset for x and y axis
    inset.set_xscale('log')
    inset.set_yscale('log')
    inset.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=3))
    inset.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=3))
    inset.tick_params(direction='in', which='both')
    inset.set_ylim([10**ylim[0], 10**ylim[1]])
    
    #inset.xaxis.set_major_locator(ticker.LinearLocator(numticks=2))
    #inset.yaxis.set_major_locator(ticker.LinearLocator(numticks=2))
    
    # Set Predict Zero Line for main graph and inset
    figaxes.axhline(1.0,  color=COLOURDICT['DATA'], lw=prediczero_lw, alpha=0.3) 
    inset.axhline(1.0,  color=COLOURDICT['DATA'], lw=prediczero_lw, alpha=0.3)
    
    # Set Labels for main graph and inset
    figaxes.set(xlabel=r'Steps Fwd [num]', ylabel=r'N. $\langle (f_n -\hat{f}_n)^2\rangle_D$ [1]')
    inset.set(xlabel=r'Steps Fwd [num]', ylabel=r'N. $\langle (f_n -\hat{f}_n)^2\rangle_D$ [1]')

    return figaxes, inset
    
##################################################################
#   NORMED BAYES RISK - COMPARE BETWEEN KALMAN X AND LSF ONLY
##################################################################


def get_Kalman_LSF_difference(kalmanX, test_case, variation, path, fstep=50, sstep=50, lowloss=20, give_ratio=0):
    
    ALGOLIST=[kalmanX, 'LSF']
    
    testobj = le(test_case, variation, 
                 AKF_path=path, AKF_load='Yes',
                 LKFFB_path=path, LKFFB_load='Yes',
                 LSF_path=path, LSF_load='Yes',
                 GPRP_path=path, GPRP_load='No')
    RISKDICT = build_risk_dict(testobj)
    
    n_train = get_n_train(testobj)
    
    normed_forecasts = [] # normed_forecasts[0] == KF data, normed_forecasts[1] == LSF data

    for algo_type in ALGOLIST:
        
        p_err, hyp, f_err, truth, lsf = RISKDICT[algo_type]

        if lsf != 0: # proceed only if data has been loaded

            if algo_type == kalmanX:
                opt_idx = analyse_kalman_errs(p_err, hyp, sstep)[0][0]
            elif algo_type =='LSF':
                opt_idx=0

            norm_f = norm_risk(f_err[opt_idx, ..., 0:fstep], 
                               truth[opt_idx, ...], 
                               n_train,
                               opt_state_err=p_err[opt_idx, ...],
                               LSF=lsf)[1] # get only forecasts from AKF or LSF
            normed_forecasts.append(norm_f)
            
    norm_bayes_risk_diff = np.array(normed_forecasts[0]) - np.array(normed_forecasts[1]) 
    
    if give_ratio != 0:
        return np.array(normed_forecasts[0])/np.array(normed_forecasts[1])
    
    return norm_bayes_risk_diff
    
    
    
##################################################################
# GENERAL
##################################################################
def set_font_sizes(ax, fsize, Fsize):
    '''explicitly sets fontsize'''
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(fsize)
    for item in ([ax.xaxis.label, ax.yaxis.label]):
        item.set_fontsize(Fsize)
    return ax


def center_y_axis(data):
    '''Returns ylim to center a plot about zero on the y-axis'''
    
    y_min = np.min(data)
    y_max = np.max(data)
    
    ylim = np.ceil(np.max(np.asarray([abs(y_min), abs(y_max)]))) # pick the largest value (y_min or y_max), round it up
    
    return ylim*1.15

def cm2inch(value):
    return value/2.54

def pts2cm(pts):
    return 0.0352778*pts

def cm2pts(cm):
    return cm/0.0352778

def cm2px(cm, mydpi):
    return 

def svg2pdf(path2file, path2output):
    ''' Converts svg --> pdf'''

    filenamesvg = str(path2file)+'.svg'
    filenamepdf = str(path2output)+'.pdf'
    cmd = 'inkscape --file={filenamesvg} --export-pdf={filenamepdf}'.format(filenamesvg=filenamesvg, filenamepdf=filenamepdf)
    os.system(cmd)

    pass


def stackNcrop(outputname, filename1, filename2, 
           right=0, left=0, top=0, bottom=0):
    '''Vertically tacks fig 1 on top of fig 2 assuming both figs have same width.
    Crops the final result if required.'''
    
    # input PDFs
    file1 = PdfFileReader(file(filename1+'.pdf', "rb"))
    file2 = PdfFileReader(file(filename2+'.pdf', "rb"))
    page = file1.getPage(0)
    page2 = file2.getPage(0)    
    
    # Output canvas dims
    total_w = page.mediaBox.getWidth() 
    total_h = page2.mediaBox.getHeight() + page.mediaBox.getHeight()

    # Uncropped Output, Stack Figs
    output = PdfFileWriter() 
    canvas = output.insertBlankPage(width=total_w, height= total_h)
    canvas.mergeTranslatedPage(page, 0, page2.mediaBox.getHeight())
    canvas.mergeTranslatedPage(page2, 0, 0)
    canvas_ = output.getPage(0) # finalises canvas
    
    # Crop 
    cropped_canvas = croppage(canvas_, right=right, left=left, top=top, bottom=bottom)
    output_c = PdfFileWriter() 
    output_c.addPage(cropped_canvas)
    
    
    # Write Cropped Output
    outputStream = file(outputname+'.pdf', "wb")
    output_c.write(outputStream)
    outputStream.close()
    
    pass


def croppage(page1, right=0, left=0, top=0, bottom=0):
    ''' PyPDF2 cropping function for stackNcrop()
    '''
    if right+left+top+bottom != 0:
        page1.mediaBox.upperRight = (page1.mediaBox.getUpperRight_x() - right, page1.mediaBox.getUpperRight_y() - top) 
        page1.mediaBox.lowerLeft = (page1.mediaBox.getLowerLeft_x() + left, page1.mediaBox.getLowerLeft_y() + bottom)
    return page1
   
def pdf2latek(outputname):
    ''' Converts pdf --> svg --> pdf for latek [NOT USED]'''

    filepdf = outputname+'.pdf'
    filesvg = outputname+'.svg'
    cmd = 'inkscape --file={filepdf} --export-plain-svg={filesvg}'.format(filesvg=filesvg, filepdf=filepdf)
    os.system(cmd)
    cmd = 'inkscape --file={filesvg} --export-pdf={filepdf} --export-latex'.format(filesvg=filesvg, filepdf=filepdf)
    os.system(cmd)
    #!inkscape --file=$filesvg --export-pdf=$filepdf --export-latex
    pass

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, name='shiftedcmap'):
    '''
    REFERENCE:  https://stackoverflow.com/questions/7404116/defining-the-midpoint-of-a-colormap-in-matplotlib
    
    Function to offset the "center" of a colormap. Useful for
    data with a negative min and positive max and you want the
    middle of the colormap's dynamic range to be at zero

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and `midpoint`.
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0. In
          general, this should be  1 - vmax/(vmax + abs(vmin))
          For example if your data range from -15.0 to +5.0 and
          you want the center of the colormap at 0.0, `midpoint`
          should be set to  1 - 5/(5 + 15)) or 0.75
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          `midpoint` and 1.0.
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.linspace(start, stop, 257)

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129, endpoint=True)
    ])

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = matplotlib.colors.LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=newcmap)

    return newcmap
