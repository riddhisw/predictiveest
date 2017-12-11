import numpy as np
from scipy.signal import butter, buttord, lfilter

# Probability of binomial weighted by cosine 

# def get_probabilites(f_0=f_0, tau_samples=tau_samples, delta_tau=delta_tau):
#     tau = delta_tau*np.arange(tau_samples)
#     return tau, 0.5*np.cos(2*np.pi*f_0*tau) + 0.5 # added some factors to shift cosine. r != autocov


# Binomial Distribution
def one_shot_msmt(n=1, p=0.5, num_samples=1):
    return np.random.binomial(n,p,size=num_samples)


# Auto-covariance function
def get_autocovariance(x):
    
    n = x.shape[0]
    diff = x - x.mean()
    lag, autocov = np.arange(n), np.correlate(diff, diff, mode = 'full')[-n:]
    divisor = np.arange(n, 0, -1)
    
    return lag, autocov, divisor

# Auto-correlation function
def get_autocorrelation(x):
    
    n = x.shape[0]
    lag, r = np.arange(n), np.correlate(x, x, mode = 'full')[-n:]
    divisor = np.arange(n, 0, -1)
    
    return lag, r, divisor


# FFT Amplitude **2
def get_pos_PSD(r, delta_t=1.0):
    n=r.shape[0]
    S =abs(np.fft.fft(r))[0:np.floor(n/2)]**2
    delf = 1.0/(delta_t*n)
    f = delf*np.arange(np.floor(n/2))
    return f, S

# Plotter function
def plot_stats(x, delta_x, ax_1, ax_2, ax_3, ax_4):


    lag, r, divr = get_autocorrelation(x)
    lagc, autocov, divc = get_autocovariance(x)

    dualr, Sr = get_pos_PSD(r/r.shape[0], delta_t=delta_x)
    dualc, Sc = get_pos_PSD(autocov/autocov.shape[0], delta_t=delta_x)

    ax_1.plot(lag, r/r.shape[0], 'o-', label='1/N')
    ax_1.plot(lag, r/divr, 'o-', alpha=0.3, label='1/n')
    ax_2.plot(lagc, autocov/autocov.shape[0], 'o-', label='1/N')
    ax_2.plot(lagc, autocov/divc, 'o-', alpha=0.3, label='1/n')

    ax_1.set(title='Auto-correlation')
    ax_2.set(title='Auto-covariance')
    ax_3.set(title='PSD')
    ax_4.set(title='FFT Ampl**2 of Auto-covariance')

    ax_3.plot(dualr, Sr, 'o-', label='1/N')
    ax_3.plot(dualr, get_pos_PSD(r/divr, delta_t=delta_x)[1], 'o-', alpha=0.4, label='1/n')
    ax_4.plot(dualc, Sc, 'o-', label='1/N')
    ax_4.plot(dualc, get_pos_PSD(autocov/divc, delta_t=delta_x)[1], 'o-', alpha=0.4, label='1/n')

    for ax in [ax_1, ax_2]:
        ax.legend(loc=2)
        
    return ax_1, ax_2, ax_3, ax_4

# low pass filter butterworth

def low_pass_filter(t, cutoff, stopfreq, gpass, gstop, data):
    
        dt = t[1]-t[0]
        Fs = 1.0/dt
        Nyq = Fs/2.0

        Wp = cutoff/Nyq # Hz in both numerator 
        Ws = stopfreq/Nyq
        
        print "Sampling rate (Hz)", Fs
        print "Desired cutoff (Hz)", cutoff
        print "Desired stopband (Hz)", stopfreq

        print "Wp Normalised Passband Frequency [dimless]", Wp
        print "Ws Normalised Stopband Frequency [dimless]", Ws
        
        [n, Wn] = buttord(Wp, Ws, gpass, gstop)
        
        print "order", n
        print "Wn (filter gain drops) (rad)", Wn
        
        [b, a] = butter(n, Wn)
        
        print "Numerator (b) polynomials", b
        print "Denominator (a) polynomials", a
        
        data = lfilter(b, a, data)
        
        return data, b, a

# low pass filter DFT

def set_fft_zero(original_sig, timestep):
    
    FFT = np.fft.fft(original_sig/original_sig.shape[0])
    freq = np.fft.fftfreq(original_sig.shape[0], d=timestep)

    FFT_f = np.zeros(FFT.shape, dtype=complex)
    FFT_f[:] = FFT[:]
    FFT_f[ int(FFT.shape[0]/4): int(FFT.shape[0]/2)] = 0
    FFT_f[int(FFT.shape[0]/2):int(FFT.shape[0]/2) + int(FFT.shape[0]/4)] = 0
    
    filtered_signal = np.fft.ifft(FFT_f)*original_sig.shape[0]
    
    return filtered_signal
