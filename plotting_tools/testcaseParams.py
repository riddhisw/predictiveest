########################################
# REFERENCE PARAMETERS (NO CHANGE) 
########################################

ADD_LS_DATA='Yes'
DO_SKF='No'
max_stp_fwd=[]

stps_fwd_truncate_=50 # Number of time steps in forecasting zone to plot
TRUNCATION = 20 # X lowest Loss values to label as being 'optimal', where X = TRUNCATION 
kea_max = 10**3

max_forecast_loss_list = [50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50]
skip_list = [1, 1, 1 , 1 , 1 , 1, 1, 1, 1, 1, 1, 1, 1]
skip_list_2 = [0, 1, 2, 3, 4, 5, 10, 16]

Hard_load='No' 
SKF_load='No'

## Amplitudes
FUDGE = 0.5
HILBERT_TRANSFORM = 2.0

## Kalman Basis
BASIS_PRED_NUM = 0 # or 1 for Basis A



