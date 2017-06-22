import kf.fast_2 as Kalman
import numpy as np

from kf.common import calc_inst_params
from kf.armakf import autokf as akf
from analysis_tools.common import calc_AR_PSD

FUDGE = 0.5
HILBERT_TRANSFORM = 2.0
################
# LSF
################

def LSF_run(LdExp, y_signal, path_to_statePredictions='./LS_Ensemble_Folder/'):

    import sys
    sys.path.append(path_to_statePredictions)
    import statePredictions as sp
    
    if LdExp.LSF_load != 'No':

        weights = np.mean(LdExp.LSF_macro_weights[:,:,:,0],  axis=0) # Dims: ensemble x stpsfwd x pastmsmts x 1
        n_predict = weights.shape[0]
        order = weights.shape[1]

        predictions = np.zeros(n_predict)

        for idx_steps in range(0, n_predict, 1):

            validation_data = sp.build_training_dataset(y_signal, 
                                                        past_msmts=order,
                                                        steps_forward=idx_steps, # testing data for n-step ahead
                                                        steps_between_msmts=1)
            
            past_measurements = validation_data[:,1:]
            actual_values = validation_data[:,0]
            predictions[idx_steps] = sp.get_predictions(weights[idx_steps,:], past_measurements)[LdExp.LSF_n_start_at] # prediction post n_train at idx_steps ahead

        return predictions


################
# AKFB 
################

def AKF_run(LdExp, y_signal, **kwargs):

    if LdExp.AKF_load != 'No':

        oe=0.0
        rk=0.0

        if len(kwargs) == 2:
            oe = kwargs['opt_sigma'] # Optimally tuned
            rk = kwargs['opt_R'] # Optimally tuned
        
        weights = LdExp.AKF_weights
        order = weights.shape[0]

        akf_pred = akf('AKF', y_signal, weights, oe, rk, 
                        n_train=LdExp.Expt.n_train, 
                        n_testbefore=LdExp.Expt.n_testbefore, 
                        n_predict=LdExp.Expt.n_predict, 
                        p0=LdExp.LKFFB_kalman_params[3], # same as LKFFB p0
                        skip_msmts=1,  
                        switch_off_save='Yes')
        
        akf_x, akf_y = calc_AR_PSD(weights, oe, LdExp.Expt.Delta_S_Sampling, LdExp.Expt.Delta_T_Sampling)


        LdExp.Truth.beta_z_truePSD() # new line. If beta_z_truePSD() is not called, true_S_norm = None
        akf_y_norm = akf_y*1.0/LdExp.Truth.true_S_norm 

        print('Total coeff', np.sum(akf_y), np.sum(akf_y_norm))

        return akf_x, akf_y, akf_y_norm, akf_pred 


################
# GPRP
################

def choose_GPR_params(LdExp):

    from plotting_tools.risk_analysis import sort_my_vals

    loss = np.mean(LdExp.GPRP_GPR_PER_prediction_errors, axis=1)
    trials = LdExp.GPRP_GPR_PER_prediction_errors.shape[0]

    idxp, periods = sort_my_vals(LdExp.GPRP_GPR_opt_params[:, 2])
    idxs, vals = sort_my_vals(loss)

    for increment in range(1, trials, 2):
        try:
            ok_losses = set(idxs[0:increment])
            ok_periods = set(idxp[trials-increment:])
            ideal_instances = list(ok_losses.intersection(ok_periods))

            if len(ideal_instances) >=1 and increment <=4:
                first_instance = ideal_instances[0]
                print('First Few Ideal Instance:', ideal_instances, ' in bottom (top) %s loss values (periodicity values)' %(increment))

            if len(ideal_instances) >=10:
                print('Ideal Instances:', ideal_instances, ' in bottom (top) %s loss values (periodicity values)' %(increment))
                break
        except:
            print("boo")
            continue
        
    try:
        ans = LdExp.GPRP_GPR_opt_params[first_instance, :]
    except:
        ans = LdExp.GPRP_GPR_opt_params[ideal_instances[0], :]
    
    return ans


def GPRP_run(LdExp, y_signal):

    import GPy 

    # Create training data objects and test pts for GPy
    X = LdExp.Expt.Time_Axis[0:LdExp.Expt.n_train, np.newaxis]
    Y = y_signal[0:LdExp.Expt.n_train, np.newaxis]
    testx = LdExp.Expt.Time_Axis[LdExp.Expt.n_train - LdExp.Expt.n_testbefore : ]

    #Set Chosen Params for GPR Model
    if LdExp.GPRP_load != 'No':
        
        R_0, sigma_0, period_0, length_scale_0 = choose_GPR_params(LdExp) 

        kernel_per = GPy.kern.StdPeriodic(1, period=period_0, variance=sigma_0, lengthscale=length_scale_0)
        gauss = GPy.likelihoods.Gaussian(variance=R_0)
        exact = GPy.inference.latent_function_inference.ExactGaussianInference()
        m1 = GPy.core.GP(X=X, Y=Y, kernel=kernel_per, likelihood=gauss, inference_method=exact)

        # Predict
        predictions = m1.predict(testx[:,np.newaxis])[0].flatten()

        return predictions

################
# LKFFB
################

def LKFFB_amps(LdExp, freq_basis_array=None, instantA=None):

    """ Returns LKFFB estimates of amplitudes in one run and theoretical PSD"""


    LdExp.Truth.beta_z_truePSD()

    if (freq_basis_array == None) and (instantA == None):
        freq_basis_array = LdExp.Truth.freq_basis_array 
        instantA = LdExp.Truth.instantA

    x_data = [2.0*np.pi*freq_basis_array, LdExp.Truth.true_w_axis[LdExp.Truth.J -1:]]

    kalman_amps = (instantA**2)*(2*np.pi)*FUDGE
    theory_PSD = HILBERT_TRANSFORM*LdExp.Truth.true_S_twosided[LdExp.Truth.J -1:]

    norm_kalman_amps = kalman_amps*(1.0/ LdExp.Truth.true_S_norm)
    norm_theory_PSD = theory_PSD*(1.0/ LdExp.Truth.true_S_norm)

    y_data = [norm_kalman_amps, norm_theory_PSD]

    return x_data, y_data, [np.sum(kalman_amps), LdExp.Truth.true_S_norm]


def LKFFB_run(LdExp, y_signal, **kwargs):      
    
    oe=0.0
    rk=0.0
    method='ZeroGain'

    if len(kwargs) == 2:
        oe = kwargs['opt_sigma'] # Optimally tuned
        rk = kwargs['opt_R'] # Optimally tuned
    
    x0 = LdExp.LKFFB_kalman_params[2]
    p0 = LdExp.LKFFB_kalman_params[3]
    bdelta = LdExp.LKFFB_kalman_params[4]

    freq_basis_array = np.arange(0.0, LdExp.Expt.bandwidth, bdelta)

    predictions, x_hat = Kalman.kf_2017(y_signal, 
                                        LdExp.Expt.n_train, 
                                        LdExp.Expt.n_testbefore, LdExp.Expt.n_predict, 
                                        LdExp.Expt.Delta_T_Sampling, 
                                        x0, p0, oe, rk, freq_basis_array, 
                                        phase_correction=0 ,prediction_method=method, 
                                        skip_msmts=1, switch_off_save='Yes')

    x_hat_slice = x_hat[:,:, LdExp.Expt.n_train]
    instantA, instantP = calc_inst_params(x_hat_slice)

    x_data, y_data, LdExp.Truth.true_S_norm = LKFFB_amps(LdExp, freq_basis_array=freq_basis_array, 
                                                                instantA=instantA)
    return x_data, y_data, LdExp.Truth.true_S_norm, predictions # this has theory and KF data
    

TUNED_RUNS_DICT = {}
TUNED_RUNS_DICT['LSF'] = LSF_run
TUNED_RUNS_DICT['AKF'] = AKF_run
TUNED_RUNS_DICT['GPRP'] = GPRP_run
TUNED_RUNS_DICT['LKFFB'] = LKFFB_run
