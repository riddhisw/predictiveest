# The purpose of this script is to write an autoregressive process order (p) 
# in a state space model, and implement Kalamn Filtering for a tuned (known) set of 
# weights. 

import numpy as np
import sys
sys.path.append('../')

from kf.common import calc_residuals, calc_Kalman_Gain

def get_autoreg_model(order, weights):
    """ Returns the dynamic state space model based 
    on order of autoregressive process and time invariant tuned weight.

    a: [Dim: order x order. dtype=float64]
    order:[Dim:  1x1. dtype=int]
    weights = [Dim: 1 x order. dtype=float64]

    """
    a = np.zeros((order, order))

    # Allocate weights
    a[0,:] = weights
    # Pick off past values 
    idx = range(order-1)
    idx2 = range(1, order, 1)
    a[idx2,idx] = 1.0

    return a 


def propagate_states_no_gamma(a, x_hat, P_hat, Q):

    '''Returns state propagation without a Kalman update and no adaptive noise 
    features (Gamma). 
    '''
    x_hat_apriori = np.dot(a, x_hat) 
    #print(a, x_hat, x_hat_apriori)

    P_hat_apriori = np.dot(np.dot(a,P_hat),a.T) + Q

    #print('Apriori x_hat 3', x_hat_apriori.shape)
    #print('Apriori P_hat 3', P_hat_apriori.shape)
    return x_hat_apriori, P_hat_apriori


def autokf(descriptor, y_signal, weights, oe, rk, n_train=1000, n_testbefore=50, 
           n_predict=50, p0=10000, skip_msmts=1):

    num = y_signal.shape[0]
    order = weights.shape[0]
    
    e_z = np.zeros(num)

    idx = range(order)
    h = np.zeros(order)
    P_hat = np.zeros((order, order))
    x_hat_apriori = np.zeros((order,1)) 
    x_hat = np.zeros((order,1))

    #print('Apriori x_hat 1', x_hat.shape)
    #print('Apriori P_hat 1', P_hat.shape)

    Q = oe*np.eye(order)
    a = get_autoreg_model(order, weights)
    #print('a', a, a.shape)
    h[0] = 1.0
    #print('h', h, h.shape)

    x_hat[:,0] = y_signal[0:order]
    P_hat[idx, idx] = p0

    #print('Apriori x_hat 2', x_hat.shape)
    #print('Apriori P_hat 2', P_hat.shape)

    store_x_hat = np.zeros((order,1,num))
    store_P_hat = np.zeros((order,order,num))
    store_x_hat[:,:, order] = x_hat
    store_P_hat[:,:, order] = P_hat  

    #print('Apriori x_hat 2b', x_hat.shape)
    #print('Apriori P_hat 2b', P_hat.shape)
    
    store_W = np.zeros((order,1,num)) 
    store_S_Outer_W = np.zeros((order,order,num))
    store_Q = np.zeros((order,order,num))
    store_S = np.zeros((1,1,num))

    predictions = np.zeros(n_testbefore + n_predict)


    # Start Filtering
    k = order # wait until order number of msmts have been made
    while (k< num): 
        #print k
        x_hat_apriori, P_hat_apriori = propagate_states_no_gamma(a, x_hat, P_hat, Q)
        
        if k> (n_train):
            # This loop is equivalent to setting the gain to zero (forecasting)
            x_hat = x_hat_apriori
            store_x_hat[:,:,k] = x_hat
            P_hat = P_hat_apriori
            store_P_hat[:,:,k] = P_hat
            k = k+1 
            continue 
        
        W_, S = calc_Kalman_Gain(h, P_hat_apriori, rk) # W needs to be reshaped
        W = W_.reshape(order,1)

        store_S[:,:, k] = S

        #print('Gain ', W.shape)
        #print('S', S.shape, S)
        
        #Skip msmts        
        if k % skip_msmts !=0:
            W = np.zeros((order, 1))
            
        e_z[k] = calc_residuals(h, x_hat_apriori, y_signal[k])
        #print('Residuals', e_z[k].shape)
        
        #print('Apriori x_hat', x_hat_apriori.shape)
        #print('Apriori P_hat', P_hat_apriori.shape)
        inter = W*e_z[k]
        #print('inter', inter.shape)

        x_hat = x_hat_apriori + W*e_z[k]
        store_S_Outer_W[:,:,k] = S*np.outer(W,W.T)
        P_hat = P_hat_apriori - S*np.outer(W,W.T) #Equivalent to outer(W, W)

        #print('x_hat', x_hat.shape)
        #print('P_hat', P_hat.shape)
        
        store_x_hat[:,:,k] = x_hat
        store_P_hat[:,:,k] = P_hat         
        store_W[:,:,k] = W
        
        k=k+1

    np.savez(descriptor, descriptor='AKF_'+descriptor,
        y_signal=y_signal,
        order= order, 
        x_hat=store_x_hat, 
        P_hat=store_P_hat, 
        a=a,
        h=h,
        weights=weights,
        e_z=e_z,
        W=store_W,
        Q=store_Q,
        S=store_S,
        oe=oe, 
        rk=rk,
        n_train=n_train,
        n_predict=n_predict,
        n_testbefore=n_testbefore,
        skip_msmts=skip_msmts)
    
    return store_x_hat, store_P_hat, store_W, store_S

