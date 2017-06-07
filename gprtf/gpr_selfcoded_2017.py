import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

'''
OUTDATED CODE. TO BE REPLACED BY GPRTF PACKAGE
'''

'''
References:
[1] C.E. Rasmussen and K.I Williams, 'Gaussian Processes for Machine Learning'. MIT Press, 2006.
[2] Solin, A. and Sarkka, S. 'Explicit Link Between Periodic Covariance Functions and State Space Models.' Conference Proceedings (2014)
available here: http://www.jmlr.org/proceedings/papers/v33/solin14.pdf
'''

def is_pos_def(x):
    #print np.linalg.eigvals(x)
    return np.all(np.linalg.eigvals(x) > 0) #postive definite

def is_pos_sem_def(x):
    #print np.linalg.eigvals(x)
    return np.all(np.linalg.eigvals(x) >= 0) #postive semi definite
    
def invertA(A):
    '''
    Returns inverse of A using Cholesky Factors and Least Squares (allegedly faster than inverting A using np.linalg.inv)
    '''
    L = np.linalg.cholesky(A) # Faster than inverting, and will return error if A is not a positive semi definite
    return L, np.linalg.lstsq(L.T,np.linalg.lstsq(L,np.eye(np.shape(A)[0]))[0])[0]

class gpr:
    
    '''Returns instance of Gaussian Process Regression (GPR)
    
    Keyword Arguments:
    -----------------
    kernel_type -- Type of kernel used for instance of GPR
        RQ - Rational Quadratic 
        RBF - Radial Basis Function (Squared Exponential)
        Periodic - Periodic (Sine Squared Exponential)
        NN - Neural Network
    
    params -- Kernel parameters and noise parameters [dim: 4x1; dtype: float64], defined below:
    
        Kernel: "RQ"
            params[0] = beta -- variance multipler of stochastic process
            params[1] = l -- length scale
            params[2] = alpha -- scale mixture parameter
            
        Kernel: "RBF"
            params[0] = beta -- variance multipler of stochastic process
            params[1] = l -- length scale
            
        Kernel: "Periodic"
            params[0] = beta -- variance multipler of stochastic process
            params[1] = l -- length scale
            params[2] = p -- fundamental periodicity
            
        Kernel: "NN" NOT DONE
            params[0] = beta -- variance multipler of stochastic process
            params[1] = sigma_0 -- Variance on 1st diagonal element of a neural network variance matrix for u
            params[2] = sigma -- Variance on all diagonal elements, except the first element, of a neural network variance matrix for u
    
        Measurment Noise Variance
            params[3] = measurement noise strength for Gaussian, uncorrelated noise [scalar,dtype=float64]
    
    X -- training data input points
    X_predict -- test data input point
    Y -- training data observations (msmt record)
    Means[0] - Mean of stochastic process, mu_f
    Means[1] - Mean of measurement noise, mu_er
    
    Note: Mean[0] + Mean[1] = E[P(Y)]
    
    '''
    
    #plt.rcParams["figure.figsize"] = (16, 8)
    
    def __init__(self,kernel_type,params,X,Y,X_predict,Means=[0.0,0.0]):
        
        self.kernel_type = kernel_type
        self.params = params
        self.X = X
        self.X_predict = X_predict
        self.Y = Y
        self.mu_f = Means[0]*np.ones(np.shape(self.Y)[0])
        self.mu_f_predict = Means[0]*np.ones(np.shape(self.X_predict)[0])
        self.mu_er = Means[1]*np.ones(np.shape(self.Y)[0])
        self.mu_y = self.mu_f + self.mu_er
        self.predictive_cov = np.empty([len(X_predict),len(X_predict)])
        self.predictive_mean = np.empty(len(X_predict))

        
    '''List of Covariance Functions'''

    
    def RQ(self, element1,element2):
        
        '''
        Returns element of covariance matrix based on covariance function, R(v)
        R(v) = (beta**2)*(1 + (v**2)/(2.0*(l**2)*alpha))**(-alpha)
        
        Keyword Arguments:
        ------------------
        element1 = data element of X or X_predict
        element2 = data element of X or X_predict
        v = element1 - element2
        beta = params[0] (covariance strength)
        l = params[1] (length scale parameter)
        alpha = params[2] (scale mixture)
        
        As alpha --> inf, RQ reduces to the RBF Kernel.
        
        Reference: (4.19) in [1] for non unit beta.
        '''
        
        cov_element = (self.params[0]**2)*(1 + ((element1-element2)**2)/(2.0*(self.params[1]**2)*self.params[2]))**(-self.params[2])
        
        if not np.isfinite(cov_element):
            print "cov element for element1 = %s, element2 = %s is not finite" %(element1,element2)
            
        return cov_element
        
        
    def RBF(self,element1,element2):
        '''
        Returns element of covariance matrix based on covariance function, R(v)
        R(v) = (beta**2)*exp(-v**2/(2.0*l**2))
        
        Keyword Arguments:
        ------------------
        element1 = data element of X or X_predict
        element2 = data element of X or X_predict
        v = element1 - element2
        beta = params[0] (covariance strength)
        l = params[1] (length scale parameter)
        
        Reference: (4.9) in [1] for non unit beta.
        
        '''
        cov_element = (self.params[0]**2)*np.exp(-(element1-element2)**2/(2.0*self.params[1]**2))
        
        if np.isfinite(cov_element)=='False':
            print "cov element for element1 = %s, element2 = %s is not finite" %(element1,element2)
            
        return cov_element
    
    
    def Periodic(self,element1,element2):
        '''
        Returns element of covariance matrix based on covariance function, R(v)
        R(v) = (beta**2)*exp(-(2.0/l**2)*(sin((element1-element2)*pi/p))**2)
        
        Keyword Arguments:
        ------------------
        element1 = data element of X or X_predict
        element2 = data element of X or X_predict
        v = element1 - element2
        beta = params[0] (covariance strength)
        l = params[1] (length scale parameter)
        p = params[2] (periodicity parameter)
        
        As sine^2 term --> 0, Periodic Kernel reduces to the RBF Kernel to first order Taylor approximation
        
        Reference: (23) in [2] for omega_0 = 2*pi/p, where p = periodicity.
        
        '''
        cov_element = (self.params[0]**2)*np.exp(-(2.0/self.params[1]**2)*(np.sin((element1-element2)*np.pi/self.params[2]))**2)
        
        if np.isfinite(cov_element)=='False':
            print "cov element for element1 = %s, element2 = %s is not finite" %(element1,element2)
            
        return cov_element
    
    
    def NN(self,element1,element2):
        print "This kernel is not yet defined"
        print "whoops"
    
    
    kernellabel = {'RQ': RQ ,'RBF': RBF, 'Periodic':Periodic, 'NN':NN}
      
    
    def K(self,var1,var2):
        '''
        Returns the Gram K Covariance Matrix for var1 and var2, with elements defined by R(v)
        '''
        
        if isinstance(var1, (int, float)) and isinstance(var2, (int, float)):
            n1 = 1
            n2 = 1
            
            Kmatrix = self.kernellabel[self.kernel_type](self,var1,var2) 
            #print Kmatrix
            
        else:
            n1 = var1.shape[0]
            n2 = var2.shape[0]

            Kmatrix = np.zeros((n1,n2))
            for ind1 in xrange(n1):
                for ind2 in xrange(n2):
                    Kmatrix[ind1,ind2] = self.kernellabel[self.kernel_type](self,var1[ind1],var2[ind2]) 
            #print Kmatrix
        
        return Kmatrix
    
    def Pyvariance(self):
        ''' Returns the variance of the marginal distribution Py in P(f*,y)
        
        References: 
        (a) Given by addition of Gaussian random variables {f} and errors {e}
        (b) OR, use Gaussian identities (A.5) and (A.6) on equation (2.18) of [1]
        '''
        return (self.K(self.X,self.X) + ((self.params[3]**2)*np.eye(len(self.X))))
    
    def logmarginallikelihood(self,logvalonly='True'):
        '''
        Returns:
        If logvalonly == 'True', the negative log marginal likelihood value (minimising -logmarginallikelihood = maximisation of logmarginallikelihood)
        If logvalonly == 'False', the negative log marginal likelihood value; and helper variables (helper_A, helper_Ainv) for predictive distribution
        
        References:
        Log p(y|f) reduces to (2.30) in [1] for zero mean {y}.
        Note norm of matrix can be calculated via Chowlesky decomposition factors L. 
        '''
        helper_A = self.Pyvariance()
        L, helper_Ainv = invertA(helper_A)
        log = -0.5*np.linalg.multi_dot([(self.Y - self.mu_y).T,helper_Ainv,(self.Y-self.mu_y)]) - 0.5*2.0*np.log(np.diag(L)).sum() - (helper_A.shape[0]/ 2.0)*np.log(2*np.pi)
        #log = -0.5*np.dot(np.dot(ydata_noisy.T,helper_Ainv),ydata_noisy) - 0.5*np.log(np.diag(helper_A)).sum() - (helper_A.shape[0]/ 2.0)*np.log(2*np.pi) ## incorrect second term
        
        if (logvalonly=='True'):
            return -log #returns negative log value such that this function can be minimised (log marginal likelihood can be maximised)
        return helper_Ainv,-log
    
    def predictive_dist_stats(self):
        '''Returns mean and covariance of the predictive distribution P(f*|y)
        
        References: (2.23) and (2.24) in [1] for zero mean {y}.
        
        '''
        helper_Ainv = self.logmarginallikelihood(logvalonly='False')[0]
        helper_beta = np.linalg.multi_dot([self.K(self.X_predict,self.X),helper_Ainv])
        y_predict = self.mu_f_predict + np.linalg.multi_dot([helper_beta, self.Y - self.mu_y])
        y_predict_cov = self.K(self.X_predict,self.X_predict) + np.linalg.multi_dot([helper_beta, self.K(self.X,self.X_predict)])
        
        self.predictive_mean = y_predict
        self.predictive_cov = y_predict_cov
        
        return y_predict, y_predict_cov
    
    def joint_dist_stats(self):
        '''Returns mean and covariance of the joint distribution P(f*,y)
        References: (2.21) in [1] for zero mean {y}.
       '''
        
        mean = np.concatenate((self.mu_y,self.mu_f_predict),axis=0)
        
        y_cov = self.Pyvariance() 
        fpred = self.K(self.X_predict,self.X_predict)
        cross = self.K(self.X,self.X_predict)
        
        K = np.zeros([len(y_cov)+len(fpred),len(y_cov)+len(fpred)])

        K[0:len(y_cov),0:len(y_cov)] = y_cov
        K[len(y_cov):,len(y_cov):] = fpred
        K[len(y_cov):,0:len(y_cov)] = cross.T
        K[0:len(y_cov),len(y_cov):] = cross
        
        #print 'Joint stat shapes',np.shape(mean), np.shape(K)
        return mean, K

    '''Plotting Functions'''
    
    def plot_Rv(self,var1,var2,xsize,ysize,fftcut,plot='yes'):
        '''
        Returns Covariance Function and Power Spectral Density
        '''
        Rv = self.K(var1,var2)[0,:]
        v = abs(var2-var1[0])
        
        if plot=='no':
            return v, Rv
        
        fig = plt.figure(figsize=(xsize,ysize))
        ax = fig.add_subplot(2, 1, 1)
        ax.plot(v, Rv,'o')
        s = "Process-Noise: %s; Msmt-Noise: %s; Length-Scale: %s; Other-Params: %s" %(self.params[0],self.params[3],self.params[1],self.params[2] )
        ax.text(0,1.5*self.params[0]+0.07,s,horizontalalignment='left',verticalalignment='top')
        ax.set_title("Covariance Function (%s Kernel)" %(self.kernel_type))
        ax.set_ylabel("R(v)")
        ax.set_xlabel("v")
        ax.set_xlim(xmin=0)
        ax.set_ylim(ymin=0, ymax=(1.5*self.params[0]+0.1))
        
        
        ax2 = fig.add_subplot(2, 1, 2)
        ax2.plot(np.fft.fftfreq(len(Rv),v[1]-v[0])[0:fftcut], (abs(np.fft.fft(Rv))**2)[0:fftcut],'o')
        ax2.set_title("DFT of Covariance Function (%s Kernel)" %(self.kernel_type))
        ax2.set_ylabel("Power Spectrum for R(v)")
        ax2.set_xlabel("Fourier Domain Frequency")
        #ax2.set_xlim(xmin=0)
        ax2.set_ylim(ymin=0)
        
        return fig, ax, ax2, v, Rv
    
    def plot_logmap(self,l_scan, p_scan,msmtnoise_scan, xsize,ysize, plot='yes',noise_slice=0):
              
        L,P = np.meshgrid(l_scan,p_scan)
        points_in_P = len(p_scan)
        points_in_L = len(l_scan)
        points_in_R = len(msmtnoise_scan)
        
        params_initial = np.zeros(4)
        params_initial[0:4] = self.params #saves original instance parameters
        
        Z=np.zeros([points_in_P,points_in_L,points_in_R])
        for n3 in xrange(points_in_R):
            for n1 in xrange(points_in_P):
                for n2 in xrange(points_in_L):
                    self.params[1] = L[n1,n2]
                    self.params[2] = P[n1,n2]
                    self.params[3] = msmtnoise_scan[n3]
                    #print "Parameters are:", self.params
                    #print "Initial Parameters are:", params_initial
                    Z[n1, n2,n3] = -1.0*self.logmarginallikelihood(logvalonly='True') 
                    #-1.0 mupltication converts -log to log such that maximal value is the maximal log marginal likelihood (MAP) estimate
        self.params = params_initial #restores original instance parameters
        
        if plot=='no':
            return L,P,Z
        
        fig = plt.figure(figsize=(xsize,ysize))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('Log Marginal Likelihood (%s Kernel, Process Noise: %s, Msmt Noise: %s)'%(self.kernel_type,self.params[0],self.params[3]))
        if (self.kernel_type == 'Periodic') :
            ax.set_ylabel("Periodicity")
        if (self.kernel_type == 'RQ') :
            ax.set_ylabel("Scale Mixture (alpha)")
        ax.set_xlabel("Length Scale (l)")
        i = ax.pcolormesh(L,P,Z[:,:,0],cmap='viridis')
        ax.set_xticklabels(l_scan)
        ax.set_xticks(l_scan)
        ax.set_yticklabels(p_scan)
        ax.set_yticks(p_scan)
        fig.colorbar(i)
        
        #print L
        #print P
        #print Z[:,:,0]
        return fig, ax, L,P,Z

    def plot_dist(self, name, samples,xsize,ysize):
        
        probdist = {'Apriori': ['P_{f}(n)','f(n)'],'Joint': ['P_{f*,y}(k)','k'], 'Marginal': ['P_{y}(n)','y(n)'], 'Predictive': ['P_{f*|y}(m)','f*|y(m)']}
        
        fig = plt.figure(figsize=(xsize,ysize))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title('%s Distribution'%(name))

        if name=='Apriori':
            covariance =  self.K(self.X,self.X)
            print 'Apriori Covariance PSD:', is_pos_sem_def(covariance)
            sample_vector = np.random.multivariate_normal(self.mu_f, covariance,(samples))
            mean_dist = self.mu_f
            x = np.linspace(0,len(mean_dist)-1, len(mean_dist))
      
        if name=='Marginal':
            covariance =  self.Pyvariance()
            print 'Marginal Covariance PSD:', is_pos_sem_def(covariance)
            sample_vector = np.random.multivariate_normal(self.mu_y,covariance,(samples))
            mean_dist = self.mu_y
            x = np.linspace(0,len(mean_dist)-1, len(mean_dist))
            
        if name=='Predictive':
            mean_dist, covariance = self.predictive_dist_stats()
            print 'Predictive Covariance PSD:', is_pos_sem_def(covariance)
            sample_vector = np.random.multivariate_normal(mean_dist, covariance,(samples))
            x = np.linspace(0,len(mean_dist)-1, len(mean_dist))

        if name=='Joint':
            mean_dist, covariance = self.joint_dist_stats()
            print 'Joint Covariance PSD:', is_pos_sem_def(covariance)
            sample_vector = np.random.multivariate_normal(mean_dist, covariance,(samples))
            x = np.linspace(0,len(mean_dist)-1, len(mean_dist))
        
        n = samples
        
        ax.set_ylabel(probdist[name][0])
        ax.set_xlabel(probdist[name][1])
        sorted_samples = np.sort(sample_vector,axis=0)
        up = sorted_samples[95*n//100,:]
        down = sorted_samples[5*n//100,:]
        ax.set_xticklabels(x)
        ax.set_xticks(x)
        ax.plot(x,mean_dist,label='Mean')
        ax.fill_between(x, down, up, alpha=0.25)
        
        return fig, ax

    '''From Sk Learn'''
    
    def do_ml_sklearn(self, param_grid_gpr, param_grid_krr, cv=5):
        '''
        Returns predictions and GPR (for param_grid_gpr) and KRR for (param_grid_krr) using GridSearchCV and built in functions on sklearn

        '''
                
        from sklearn import metrics
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Sum, Product, WhiteKernel, ExpSineSquared, RBF, RationalQuadratic, Matern
        from sklearn.model_selection import GridSearchCV
        from sklearn.kernel_ridge import KernelRidge
        
        krr = GridSearchCV(KernelRidge(),  cv=5, param_grid=param_grid_krr)
        krr.fit(self.X.reshape(-1, 1),self.Y)

        gpr = GridSearchCV(GaussianProcessRegressor(),  cv=5, param_grid=param_grid_gpr)
        gpr.fit(self.X.reshape(-1, 1),self.Y)

        return krr, gpr, krr.predict(self.X_predict.reshape(-1,1)),gpr.predict(self.X_predict.reshape(-1,1)) 
    
 
