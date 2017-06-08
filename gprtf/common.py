import numpy as np
import os

def invertA(A):
    '''
    Returns inverse of A using Cholesky Factors and Least Squares (allegedly faster than inverting A using np.linalg.inv)
    '''
    L = np.linalg.cholesky(A) # Faster than inverting, and will return error if A is not a positive semi definite
    return np.linalg.lstsq(L.T,np.linalg.lstsq(L,np.eye(np.shape(A)[0]))[0])[0]

def get_data(test_case, variation, points=200, randomize='y'):

    pathtodir = './LS_Ensemble_Folder'#
    filename = 'test_case_'+str(test_case)+'_var_'+str(variation)+'_BR_AKF_MAP.npz'
    datafile = os.path.join(pathtodir, filename)

    
    obj_ = np.load(datafile)

    truth = obj_['macro_truth'][1,0,:]
    n_predict = obj_['akf_macro_forecastng_errors'][1,0,:].shape[0]
    num = truth.shape[0]
    n_train = num - n_predict
    timeaxis = np.arange(0, num, 1.0)
    msmt_noise_var = obj_['msmt_noise_variance']

    if randomize =='y':
        x =[]
        y =[]

        for index in np.random.uniform(low=0, high=n_train, size=points):
            x.append(timeaxis[index])
            y.append(truth[index]+ msmt_noise_var*np.random.randn())
    
    elif randomize != 'y':
        x = np.arange(0, n_train, 1.0, dtype=np.float32)
        y = truth[0:n_train] + msmt_noise_var*np.random.randn(n_train)
        

    testx = np.arange(n_train-50, num, 1.0, dtype=np.float32)

    print('Shapes', len(x), len(y), testx.shape)
    
    return np.asarray(x,dtype=np.float32), np.asarray(y,dtype=np.float32), testx, truth, n_predict
