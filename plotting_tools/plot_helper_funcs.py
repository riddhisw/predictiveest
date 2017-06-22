'''
Plot_helper_funcs.py is a list of functions that help produce figures 
reported in analysis. 

'''
import matplotlib.pyplot as plt, numpy as np, matplotlib

from plotting_tools.risk_analysis import build_risk_dict
#from plotting_tools.load_raw_cluster_data import LoadExperiment as le
from plotting_tools.plot_figstyle_sheet import *

##################################################################
# RISK MAP AND TRAJECTORIES
##################################################################

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
    from plotting_tools.risk_analysis import riskmapdata
    
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
    
    figaxes.plot(s_sigma[0:lowloss]**2, s_R[0:lowloss], 'o', c='tan', markersize=25, alpha=0.7)
    figaxes.plot(f_sigma[0:lowloss]**2, f_R[0:lowloss], 'o', c='cyan', markersize=15, alpha=0.7)
    figaxes.plot(s_sigma**2, s_R, 'kv', markersize=5, alpha=1.0)
    figaxes.plot(s_sigma[0]**2, s_R[0],'*', color='m', markersize=15, mew=2)
    figaxes.tick_params(direction='in', which='both')
    
    figaxes.set(title=algotype, xlabel=r'$\sigma^2 \quad [f_n^2] $', ylabel =r'$R \quad [f_n^2]$')
    
    figaxes.xaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5))
    figaxes.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=5))
    
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
        
    from plotting_tools.risk_analysis import riskmapdata
    
    p_err, hyp, f_err = RISKDICT[algotype][0:3]
    s_traj, f_traj = riskmapdata(p_err, 
                                 f_err, 
                                 hyp, 
                                 maxforecaststps=fstep, 
                                 maxstateeststps=sstep)[6:8]

    for idx_traj in range(len(s_traj)-1, -1, -1):
        
        if idx_traj < lowloss:
            figaxes.plot(range(-sstep, 0, 1), s_traj[idx_traj][0:fstep], '-', markersize=4, alpha=0.4, c='tan')
            figaxes.plot(range(0, fstep, 1), f_traj[idx_traj][0:fstep], '-', markersize=4, alpha=0.4, c='c')
            
        elif idx_traj >= lowloss: 
            figaxes.plot(range(-sstep, 0, 1), s_traj[idx_traj][0:fstep], '-', markersize=4, alpha=0.1, c='k')
            figaxes.plot(range(0, fstep, 1), f_traj[idx_traj][0:fstep], '-', markersize=4, alpha=0.1, c='k')

    figaxes.plot(range(-sstep, 0, 1), s_traj[0], '-', markersize=4, c='m') 
    figaxes.plot(range(0, fstep, 1), f_traj[0][0:fstep], '-', markersize=4, c='m')
    
    
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
    figaxes.set(title=algotype, xlabel=r'Time Stps [num]', ylabel=r'log($\langle (f_n -\hat{f_n})^2 \rangle_D$) [log($f_n^2$)]')
    
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
                            fstep=50, sstep=50, lowloss=20, lgd_loc=4,
                            ylim_amps = [-4.5, 1], yscale='linear', true_undersampl_scale=0):
    '''Returns single predictions from all algorithms, spectral estimates from AKF/LSF, LKKFB'''
    
    # import FIGSTYLES
    from plotting_tools.risk_analysis import build_risk_dict, norm_risk, riskmapdata, analyse_kalman_errs
    from plotting_tools.tuned_run_analysis import TUNED_RUNS_DICT
    from plotting_tools.load_raw_cluster_data import LoadExperiment as le
    import numpy as np

    testobj = le(test_case, variation, 
                 AKF_path=path, AKF_load=AKF_load,
                 LKFFB_path=path, LKFFB_load=LKFFB_load,
                 LSF_path=path, LSF_load=LSF_load,
                 GPRP_path=path, GPRP_load=GPRP_load)
    # make data
    pickone = int(np.random.uniform(low=0, high=49))
    truth = testobj.LSF_macro_truths[pickone, :]
    signal = testobj.LSF_macro_data[pickone, :]

    ntb = testobj.Expt.n_testbefore
    ntn = testobj.Expt.n_train
    x_axis = range(-ntb, fstep, 1)
    
    true_undersampl_const = 1.0
    if true_undersampl_scale != 0:
        true_undersampl_const = calc_undersampling(testobj)
    
    # plot data
    figaxes.plot(x_axis[ : ntb], signal[ntn - ntb : ntn], 'x', label='Data')
    figaxes.plot(x_axis, truth[ntn - ntb : ntn + fstep], 'r', label='Truth')
    
    for algo_type in ALGOLIST:
        
        print(algo_type)
        
        if algo_type == 'LKFFB' or algo_type == 'AKF':

            RISKDICT=build_risk_dict(testobj)
            opt_sigma, opt_R = analyse_kalman_errs(RISKDICT[algo_type][0],
                                                   RISKDICT[algo_type][1],
                                                   50)[2:]

            KWGS = {'opt_sigma': opt_sigma[0], 'opt_R': opt_R[0]}
            faxis, amp, norm, pred = TUNED_RUNS_DICT[algo_type](testobj, signal, **KWGS)
            
            # Plot predictions (Kalman)
            figaxes.plot(x_axis[ : ntb], pred[: ntb ], '-', x_axis[ntb: ], pred[ntb : ntb + fstep], '--',
                         c=COLOURDICT[algo_type],
                         markersize=3, label=algo_type)
            
            # Plot amplitude estimates
            if algo_type == 'LKFFB':
                
                # LKFFB
                figaxes_amps.plot(faxis[0], amp[0]*(1.0/true_undersampl_const), 'o', 
                                  c=COLOURDICT[algo_type], alpha=ALPHA_AMPS, markersize= MSIZE_AMPS,
                                  label=' T.Pow: %.1e' %(norm[0]))
                # Truth
                figaxes_amps.plot(faxis[1], amp[1], 'r-', label='T.Pow: %.1e' %(norm[1]))     
                
            else:
                truncatedata = int(faxis.shape[0]*300.0/ faxis[-1])
                figaxes_amps.plot(faxis[0: truncatedata], norm[0: truncatedata], 'o', 
                                  c=COLOURDICT[algo_type], alpha=ALPHA_AMPS, markersize= MSIZE_AMPS,
                                  label=' T.Pow: %.1e' %(np.sum(norm)))
        else:

            # Plot predictions (Non Kalman)
            pred = TUNED_RUNS_DICT[algo_type](testobj, signal)
            if algo_type == 'LSF':
                figaxes.plot(x_axis[ntb :], pred, '--', markersize=3, 
                             c=COLOURDICT[algo_type],
                             label=algo_type)
            
            elif algo_type == 'GPRP' and testobj.GPRP_load != 'No':
                figaxes.plot(x_axis[ : ntb], pred[: ntb ], '-', x_axis[ntb: ], pred[ntb : ntb + fstep ], '--',
                             markersize=3, 
                             c=COLOURDICT[algo_type],
                             label=algo_type)

    # Config x-axis for amplitudes with bandwidth assumptions
    
    bandedge = testobj.Truth.f0*(testobj.Truth.J-1)*2.0*np.pi
    compedge = testobj.Expt.bandwidth*2.0*np.pi
    figaxes_amps.axvline(x=bandedge, ls='--', c=true_bandedg_clr)#, label= 'True Band Edge')
    figaxes_amps.axvline(x=compedge, ls='--', c=lkffb_bandedg_clr)#, label= 'KF Basis Ends')
    
    # Config x axis for predictions
    figaxes.axvspan(-sstep,0, color='gray', alpha=0.1)
    figaxes.set_xlim([-sstep - 10, fstep])  # start 10 points before sstep, add n_train label
    xtickslabels =[x.get_text() for x in figaxes.get_xticklabels()]
    xtickslabels[0] = str(r'$-n_{T}$')
    xtickvalues = [int(x) for x in figaxes.get_xticks()]
    xtickslabels[1:] = xtickvalues[1:]
    figaxes.set_xticklabels(xtickslabels)
    figaxes.axvspan(-sstep-10, -sstep, color='gray', alpha=0.2)
    
    # Config y axis for amplitudes
    figaxes_amps.set_yscale('log')
    figaxes_amps.set_ylim([10**ylim_amps[0], 10**ylim_amps[1]])
    
    # Config y axis for predictions
    figaxes.ticklabel_format(style='sci', scilimits=(0,2), axis='y')
    
    # Set labels
    figaxes_amps.legend(loc=lgd_loc)
    figaxes_amps.set(xlabel=r'$\omega$ [rad]', ylabel=r'$S(\omega)$ [$f_n^2$/(rad $s^{-1}$)]')
    figaxes.set(xlabel='Time Stps [num]', ylabel=r'Predictions [$f_n$]')
    
    return figaxes, figaxes_amps


##################################################################
#   NORMED MEANS
##################################################################

def plot_normed_means(figaxes, inset, ALGOLIST, test_case, variation, path,
                      GPRP_load='Yes', LSF_load='Yes', AKF_load='Yes', LKFFB_load='Yes',
                      fstep=50, sstep=50, lowloss=20, 
                      ylim = [-11, 3], yscale='log'):
    
    # import FIGSTYLES
    from plotting_tools.risk_analysis import build_risk_dict, norm_risk, riskmapdata, analyse_kalman_errs
    from plotting_tools.load_raw_cluster_data import LoadExperiment as le

    testobj = le(test_case, variation, 
                 AKF_path=path, AKF_load=AKF_load,
                 LKFFB_path=path, LKFFB_load=LKFFB_load,
                 LSF_path=path, LSF_load=LSF_load,
                 GPRP_path=path, GPRP_load=GPRP_load)
    RISKDICT = build_risk_dict(testobj)

    print(testobj.test_case, testobj.variation)

    for algo_type in ALGOLIST:
        
        #print(algo_type)
        p_err, hyp, f_err, truth, lsf = RISKDICT[algo_type]

        if lsf != 0: # proceed only if data has been loaded

            opt_idx=0

            if algo_type == 'AKF' or algo_type == 'LKFFB' :

                opt_idx = analyse_kalman_errs(p_err, hyp, sstep)[0][0]

            norm_s, norm_f = norm_risk(f_err[opt_idx, ..., 0:fstep], 
                                       truth[opt_idx, ...], 
                                       testobj.Expt.n_train,
                                       opt_state_err=p_err[opt_idx, ...],
                                       LSF=lsf)
            if lsf == 'No':

                figaxes.plot(range(-norm_s.shape[0]-1,-1,1), norm_s, 
                             c=COLOURDICT[algo_type], label=algo_type+' State Est')

            figaxes.plot(range(0,norm_f.shape[0],1), norm_f, '--', 
                            c=COLOURDICT[algo_type], label=algo_type+' Forecast')

            inset.plot(range(0,norm_f.shape[0],1), norm_f, '--', c=COLOURDICT[algo_type])
        

    
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
    inset.set_yscale('linear')
    #inset.ticklabel_format(style='sci', scilimits=(0,2), axis='y')
    inset.tick_params(direction='in', which='both')
    #inset.xaxis.set_major_locator(ticker.LinearLocator(numticks=2))
    #inset.yaxis.set_major_locator(ticker.LinearLocator(numticks=2))
    
    # Set Predict Zero Line for main graph and inset
    figaxes.axhline(1.0,  color='darkblue') 
    inset.axhline(1.0,  color='darkblue')
    
    # Set Labels for main graph and inset
    figaxes.set(xlabel=r'Time Stps [num]', ylabel=r'Norm log($\langle (f_n -\hat{f_n})^2 \rangle_D$) [1]')
    inset.set(xlabel=r'T. Stps [num]', ylabel=r'N. log(BR)')

    return figaxes, inset