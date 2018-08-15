#########################################################################
#########         Optimal prediction code written by            #########
#########        Virginia Frey and Sandeep Mavadia 2016         #########
#########################################################################
# 
#########################################################################
#########           Modified by Riddhi Gupta (2017)           ###########
#########################################################################


import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.colors import LogNorm
from matplotlib import ticker

''' State prediction for FS experiments with various parameters.

Version: 0  Module: `statePredictions`
            Authors: Virginia Frey and Sandeep Mavadia
            Year: 2016

Version: 1  Module: `statePredictions_2`
            Modifed by: Riddhi Gupta
            Year: 2017
            Details: Modified function calls to access alpha-learnng hyperparameter
                     in gradient descent as an additional parameter for tuning
                     in LSF.
'''

def build_training_dataset(measured, engineered=np.array([]),
                           past_msmts=3, steps_forward=1, steps_between_msmts=1,include_offset=True):
    ''' Creates a matrix based on measured data that can then be used for linear regression via gradient
        descent.
        Input:
            measured                1D data set
            engineered              (optional) 1D dataset if we want to predict using features of a second dataset
                                    (for experimental purposes, not actually in use)
            past_msmts              The number of labels (past measurements) to be included in the calculation
            steps_forward           (optional) discrete steps forward in the data set
            steps_between_msmts     (optional) discrete steps between points in the data set
            include_offset          (optional) includes a DC offset for which a coefficient will be calculated

        Returns:
            new_dataset             A matrix whose first column contains all the labels we want to predict and
                                    the other columns contain the corresponding features
        '''

    if engineered.size != 0:
        if measured.shape == engineered.shape:      # Check that dimensions match
             predict_noise = True                   # If True: predict noise, otherwise predict next measurement
        else:
            print "In function build_training_dataset: Dimensions of the given datasets do not match."
            return
    else:
        predict_noise = False
                
                                   
    m = measured.shape[0]               # Total number of data points

    n = past_msmts                      # number of past measurements to be used for prediction
    k = steps_forward   + steps_between_msmts -1                # steps forward
    s = steps_between_msmts             # steps between measurements


    if include_offset == False:
        new_dataset = np.zeros((m-(n*s-s)-k,n+1))
    else:
        new_dataset = np.zeros((m-(n*s-s)-k,n+2))       

    for i in range(0,m-(n*s-s)-k):                              # loop through rows
        if predict_noise==True:                                 # predict noise or msmts
            new_dataset[i,0] = engineered[(n-1)*s+k+i]          # set first column value in the data matrix
        else:
            new_dataset[i,0] = measured[(n-1)*s+k+i]
        
        for j in range(1,n+1):                                  # loop through columns
            new_dataset[i,j] = measured[(n-1)*s+i-(j-1)*s]
        if include_offset == True:
            new_dataset[i,n+1] = 1   

    return new_dataset




def rms_error(actual_values,predicted_values):
    ''' Calculates the RMS error of the predicted and actual values, both given as 1D vectors '''
    squared_error = (actual_values-predicted_values)**2
    mean_squared_error = np.mean(squared_error)
    return mean_squared_error**(0.5)


def gradient_summand(weights,actual_values,past_measurements):
    ''' Calculates the gradient summand for given weights, actual values (both 1D vectors) and
        the past measurements (given as a matrix).

        weights (dx1)
        actual_values (nx1)
        past_measurements (nxd)

        gSummands is a (nxd) matrix where each row corresponds to the gradient summand
        for one particular prediction.
            '''
    gSummands = (np.dot(past_measurements,weights)- actual_values) * past_measurements
    return gSummands
    
    

def get_predictions(weights,past_measurements):
    '''  Calculates the prediction for each point in actual_values by multiplying the weights
         and the past measurements.

         Dimensions:
            weights (dx1)
            past_measurements (nxd)

         predictions is a (nx1) column vector
             ''' 
    predictions = np.dot(past_measurements,weights)
    return predictions


def gradient_descent(training_data,numIters,initial_weights=0,alpha_coeff=0.05):
    ''' Linear regression via gradient descent.

        training_data: (nxd) matrix containing where each row contains the msmt value that is
        to be reconstructed via linear combiniation of the past measurements in that row.
                    
        n: Length of training data
        d-1: Number of past measurements used for the regression

        In each iteratioin step the weights are used to calculate a prediction and the RMS error
        between predicted and actual values are saved in a list called 'errorTrain'.

        The gradient for each set of weights is calculated using 'gradient_summand(args)', which
        returns a (nxd) matrix where each row corresponds to the gradient summand for one particular
        prediciton. Summing all the rows up gives the overall gradient ((d-1)x1 vector)
        
        The weights are updated in each step by subtracting the gradient multiplied by a gain
        factor alpha.

        Function returns the weights in a column vector and the errorTrain
        '''
    n = training_data.shape[0]                              # No. of rows
    d = training_data.shape[1]                              # No. of columns
    
    actual_values = np.zeros((n,1))                         # extract actual values from the dataset
    actual_values[:,0] = training_data[:,0]                 # and save in a column vector
    past_measurements = training_data[:,1:d]                # extract past msmts used for prediction
                                                            # and save in a nx(d-1) array

    weights = np.zeros((d-1,1))                             # initial values for the weights and gradient
    #weights = np.ones((d-1,1))                                                        
    gradient = np.zeros((d-1,1))                            # weights, gradient are column vectors
    alpha = alpha_coeff                                     # gain factor

    weights[0,0] += 1                                       # same situation as in traditional feedback
    #weights *= initial_weights

    #print 'In gradient_descent:',training_data[0,:]
    errorTrain = np.zeros(numIters)
    
    no_of_repetitions = 0
    while True:

        if no_of_repetitions > 10:
            print 'WARNING: \t In gradient_descent: no convergence after 25 repetitions.'
            print 'last error_difference: '+str(error_difference)
            break
        
        for i in range(numIters):
            predicted_values = get_predictions(weights,past_measurements)   # calculate predictions with current weights
            errorTrain[i] = rms_error(actual_values,predicted_values)       # and the RMS error 


            gradient[:,0] = np.sum(gradient_summand(weights,actual_values,  
                                                past_measurements),axis=0) # Sum up all the rows in gradient_summand
                                                    
            alpha_i = alpha / (n * np.sqrt(i+1))
            weights = weights - alpha_i * gradient              # update the weights

        error_difference = abs(errorTrain[-1]-errorTrain[-2])
        if  error_difference < 0.005:
            #print 'last error_difference:',error_difference
            break
        
        if no_of_repetitions == 0:
            previous_error_difference = error_difference
            alpha *= 1.03 # increase alpha
            no_of_repetitions += 1
            print 'Gradient descent: first error_difference='+str(error_difference)
            continue
        if no_of_repetitions == 1:
            if error_difference < previous_error_difference:
                increase_alpha = True
                #print 'increase: second error_difference='+str(error_difference)
            else:
                increase_alpha = False  # we have to decrease alpha instead
                #print 'decrease: second error_difference='+str(error_difference)

        alpha = alpha*2 if increase_alpha == True else  alpha/2
            
        no_of_repetitions += 1
   
            
    return weights , errorTrain
        


    
def calculate_range_of_predictions(measured, engineered=np.array([]),
                           max_past_msmts=3, max_steps_forward=1, steps_between_msmts=1,pct=-1, validation_dataset = np.array([])):
    '''
        Creates a dataset with variable numbers of past measurements used for predictions and steps forward based
        on measured data using gradient_descent(args[]).

        For each number of past measurements and steps forward the RMS error between actual and predicted value
        is calculated and saved in a 2D array with dimensions (max_past_msmts x max_steps_forward).

        A distinction between training and validation data can be made by using 'pct' elem [0,1] ...  
        '''

    print '---------------------------------------------------------------------------'
    print 'Calculating the range of predictions for '
    print '1-'+str(max_past_msmts)+' past measurements and 1-'+str(max_steps_forward)+' steps forward.'
    print 'Steps between measurements:',steps_between_msmts
    mpm = max_past_msmts                    # rename and introduce variables
    msf = max_steps_forward
    sbm = steps_between_msmts
    numIters = 50
    use_validation_data = False             # use a certain percentage of the given dataset as validation data or
    use_ext_validation_data = False         # use a completely different dataset for validation
    m = measured.shape[0]                   # number of points in dataset

    weights = np.zeros((mpm,msf), dtype=np.ndarray)     # saves the weights for each realisation of gradient_descent
    errorTrain = np.copy(weights)                       # saves the errorTrain  - " -

    rms_training_data = np.zeros((mpm,msf))     # RMS values for the training dataset

    
    if (0.1 < pct < 0.9) and validation_dataset.size == 0:
        use_validation_data = True              # pct is fraction of training dataset / whole dataset
        n = np.floor(pct*m)                     # size of the training dataset
        print "Points in training data:",n
        print "Points in validation data:",m-n
        rms_validation_data = np.zeros((mpm,msf))   # RMS values for the validation dataset
    elif validation_dataset.size != 0:
        use_ext_validation_data = True              # use external dataset for validation
        rms_validation_data = np.zeros((mpm,msf))
        print 'Use different dataset for validation.'
        n = m
    else:
        n = m
    
    tic = time.time()
    for i in range(0,mpm):                              # loop through numbers of past measurements
        #print 'past measurements: ',i+1
        for j in range(0,msf):                          # loop through steps forward
            
            train_data = build_training_dataset(measured[0:n],              # build datasets for gradient_descent
                                                engineered[0:n],i+1,j+1,sbm)

            weights[i,j], errorTrain[i,j] = gradient_descent(train_data,numIters) # perform regression

            rms_training_data[i,j] = errorTrain[i,j][-1]            # the RMS of the training dataset is just the
                                                                    # last value in 'errorTrain'

            if abs(errorTrain[i,j][-2] - errorTrain[i,j][-1]) > 0.01:
                print 'WARNING: slow convergence for '+str(i+1)+' past msmts and '+str(j+1)+' steps forward.'

            #s = train_data.shape; d=s[1]; r=s[0]                    # number of columns and rows
            #actual_values = np.zeros((r,1))
            #actual_values[:,0] = train_data[:,0]
            #predictions = get_predictions(weights[i,j],train_data[:,1:d])         # calculate predictions and RMS
            #rms_training_data[i,j] = rms_error(actual_values,predictions)         # errors for the validation data
            
                                                                    
            if use_validation_data == True or use_ext_validation_data == True:      # Use validation data, either internal or external
                if use_validation_data == True:
                    validation_data = build_training_dataset(measured[n:m],         # Use a certain percentage of the given dataset for validation
                                                         engineered[n:m],i+1,j+1,sbm)
                else:                                                               # Use external dataset for validation
                    if len(list(validation_dataset.shape)) == 1:                    # either a pure noise trace
                        validation_data = build_training_dataset(validation_dataset,past_msmts=i+1,
                                                             steps_forward=j+1, steps_between_msmts=sbm)
                    else:                                                           # or a dataset of msmts and applied noise
                        validation_data = build_training_dataset(validation_dataset[0,:],validation_dataset[1,:],past_msmts=i+1,
                                                             steps_forward=j+1, steps_between_msmts=sbm)
                        
                    
                s = validation_data.shape; d=s[1]; r=s[0]           # number of columns and rows
                actual_values = np.zeros((r,1))
                actual_values[:,0] = validation_data[:,0]
                predictions = get_predictions(weights[i,j],validation_data[:,1:d])      # calculate predictions and RMS
                rms_validation_data[i,j] = rms_error(actual_values,predictions)         # errors for the validation data
            
    toc = time.time()-tic
    print 'Time taken to calculate the range of predictions:',np.round(toc,2),'s'
    print '---------------------------------------------------------------------------'
    if use_validation_data == True or use_ext_validation_data == True:                 # return either training or validation RMS values
        return rms_validation_data
    else:
        return rms_training_data                                      


def traditional_feedback(measured, engineered=np.array([]),
                            max_steps_forward=1, steps_between_msmts=1):
    '''
        Calculates the RMS error between actual measurements and values predicted by traditional feedback (TFB). 
        '''
    msf = max_steps_forward
    sbm = steps_between_msmts
    TFB_rms_errors = np.zeros(msf)
    n = measured.shape[0]

    if engineered.size == 0:            # predict next measurement
        engineered = np.copy(measured)
    
    for steps_forward in range(1,msf+1):      
        
        no_of_points = n - steps_forward - sbm + 1
        actual_values = np.zeros(no_of_points);
        predicted_values = np.copy(actual_values)
        for j in range(0,no_of_points):
            actual_values[j] = engineered[j+steps_forward+sbm-1]
            predicted_values[j] = measured[j]
            
        #print actual_values
        #print predicted_values
        TFB_rms_errors[steps_forward-1] = rms_error(actual_values,predicted_values)

    return TFB_rms_errors
    




def prepare_plot(data, TFB_data=np.array([]), ColourBarMin=0,ColourBarMax=0,LogScale=False):
    '''
        Prepares to plot a 2D contour plot of a dataset created by 'calculate_range_of_predictions(args[])'
        by setting variables like axes, colourmaps and co.
        The number of past measurements is on the x-axis, the number of steps forward on the y-axis and
        the colour represents the RMS error between actual and predicted values. Maximum and minimum RMS
        error can be given as optional arguments for the colourbar.
        '''
    include_TFB_data = False
    data = np.transpose(data)
    rows = data.shape[0]
    cols = data.shape[1]
    
    if TFB_data.size != 0:
        if TFB_data.shape[0] != data.shape[0]:
            print "In prepare_plot(): TFB and gradient descent data must have the same first dimension."
            return
        else:
            include_TFB_data = True
            full_data = np.zeros((rows,cols+1))
            full_data[:,0] = TFB_data[:]
            full_data[:,1:cols+1] = data
            data = full_data

    if ColourBarMin == ColourBarMax:            # If no range is given, use defaults 
        ColourBarMin = np.min(data)
        ColourBarMax = np.max(data)
    
    fig, ax = plt.subplots()                    # Main plotting commands
    #-------------------------------------------------------------------------------------------------------
    if LogScale == True:                                                    # Plot colour bar on log scale
        data_plot = ax.imshow(data, interpolation='nearest',    
                              aspect='auto', cmap=plt.cm.magma,norm=LogNorm(),
                              vmin=ColourBarMin,vmax=ColourBarMax)
        colourbar = plt.colorbar(data_plot)
        
        tick_locs   = np.round( np.logspace( np.log10(ColourBarMin), np.log10(ColourBarMax), 5  ), 4 )
        if tick_locs[0] < ColourBarMin:
            tick_locs[0] = np.round( ColourBarMin+0.05 , 4 )
        if tick_locs[-1] > ColourBarMax:
            tick_locs[-1] = np.round( ColourBarMax-0.05 , 4 )
        tick_labels = np.round(tick_locs,2)


        colourbar.locator     = ticker.FixedLocator(tick_locs)
        colourbar.formatter   = ticker.FixedFormatter(tick_labels)
    
        colourbar.update_ticks()
    else:                                                                   # Plot colour bar on normal scale
        data_plot = ax.imshow(data, interpolation='nearest',    
                              aspect='auto', cmap=plt.cm.magma,
                              vmin=ColourBarMin,vmax=ColourBarMax)
        colourbar = plt.colorbar(data_plot)
    colourbar.set_label('RMS error between predicted and actual value')
        
    #-------------------------------------------------------------------------------------------------------
                            # Configure ticks and labels for x,y-axes  
    #-------------------------------------------------------------------------------------------------------    
    x_shape = data.shape[1]
    y_shape = data.shape[0]
    
    ax.set_xticks(np.arange(x_shape))                   # put the major ticks at the middle of each cell
    ax.set_yticks(np.arange(y_shape))
    ax.axis([-0.5, x_shape-0.5, -0.5, y_shape-0.5])     # set axis limits (have .5 for nice formatting)

    x_labels = [item.get_text() for item in ax.get_xticklabels()]   # set tick labels manually

    if include_TFB_data == True:                                    
        x_labels[0] = 'SFB'
        x_labels[1:] = np.arange(1,x_shape+1,1)
    else:   
        x_labels[:] = np.arange(1,x_shape+1,1)                        
    
    y_labels = [item.get_text() for item in ax.get_yticklabels()]
    y_labels[:] = np.arange(1,y_shape + 1,1)
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)

    ax.set_xlabel('Number of past measurements')
    ax.set_ylabel('Steps forward')

    return fig,ax

    
   
def main():
    
    # junk dataset to see it's working
    measured = np.zeros(11)
    engineered = np.zeros(11)

    for i in range(0,11):
        measured[i] = i
        engineered[i] = 50 + i

    data = build_training_dataset(measured,engineered,3,1,3)
    print data

    print traditional_feedback(measured,engineered,1,3)
    

    # real dataset

    #engineered = np.loadtxt("Sep30_4_appliednoise.txt",skiprows=2)
    #measured = np.loadtxt("Sep30_4_qubit.txt",usecols=(1,))
    return


if __name__ == '__main__':
    main()





            


    

    
