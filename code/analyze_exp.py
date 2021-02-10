'''
    MI.py

    Analyze data from MI experiment
    Zeldenrust, F., de Knecht, S., Wadman, W. J., Denève, S., Gutkin, B., Knecht, S. De, Denève, S. (2017). 
    Estimating the Information Extracted by a Single Spiking Neuron from a Continuous Input Time Series. 
    Frontiers in Computational Neuroscience, 11(June), 49. doi:10.3389/FNCOM.2017.00049
    Please cite this reference when using this method.

    INPUT:
    ron, roff (kHz): switching speed of the hidden state
    x: array with hidden state values over time
    input_theory: array with unscaled input current values (output from ANN)
    dt: binsize recordings (ms)
    spiketrain: array (same size as x and input_theory) of 0 (no spike) and 1 (spike)

    OUTPUT
    Output-dictionary with keys: #TODO the spiketrain stuff
    MI_i        : mutual information between hidden state and input current
    xhat_i      : array with hidden state estimate based on input current 
    MSE_i       : mean-squared error between hidden state and hidden state estimate based on input current 
    MI          : mutual information between hidden state and spike train
    qon, qoff   : spike frequency during on, off state in spike train 
    xhatspikes  : array with hidden state estimate based on spike train
    MI          : mean-squared error between hidden state and hidden state estimate based on spike train
'''
import numpy as np
from scipy import stats, integrate

def analyze_exp(ron, roff, x, input_theory, dt, theta): #TODO add spiketrain for additional calc
    ''' Analyzes the the hidden state and the input that was created by the ANN to
        create the Output dictionary.
        Equations 13 & 14
    '''
    spiketrain = 0
    Output = {}

    # Input
    Hxx, Hxy, Output['MI_i'], L_i = calc_MI_input(ron, roff, input_theory, theta, x , dt)
    Output['xhat_i'] = 1. / (1 + np.exp(-L_i))
    Output['MSE_i'] = np.sum((x - Output['xhat_i'])**2)
    return Output


def dLdt_input(L, ron, roff, I, theta):
    ''' Differential equation calculating the posterior Log-likelihood of the 
        hidden state being 1 based on the input history.
        Equation 10.
    '''
    dLdt = ron * (1 + np.exp(-L)) - roff * (1 + np.exp(L)) + I - theta
    return dLdt

def p_conditional(L):  
    ''' Estimates the probability that the hidden state equals 1 given the input history.
        Equation 11.
    '''
    return 1 / (1 + np.exp(-L))

def MI_est(L, x):
    ''' Calculates the mutual information (MI) based on the entorpy of the hidden state (Hxx)
        and the conditional entropy of the hidden state given the input (Hxy).
        Equations 4, 6 & 8  
    '''
    Hxx = - np.mean(x) * np.log2(np.mean(x)) - (1 - np.mean(x)) * np.log2(1 - np.mean(x))
    Hxy = - np.mean(x * np.log2(p_conditional(L)) + (1 - x) * np.log2(1 - p_conditional(L)))
    MI = Hxx - Hxy
    return Hxx, Hxy, MI


def calc_MI_input(ron, roff, I, theta, x, dt):
    ''' Calculate the mutual information between hidden state x and
        generater input train (same size vector) assuming a ideal observer
        that knows ron, roff and theta.

        Note that if dt in ms, then ron and roff in kHz.
        Note that information is calculated in bits. For nats use log instead of log2.
    '''
    # Integrate the posterior Log-likelihood
    L = np.empty(len(x))
    L[0] = np.log(ron/roff)
    for i in range(len(x) - 1):
        L[i + 1] = L[i] + dLdt_input(L[i], ron, roff, I[i], theta) * dt
        if abs(L[i+1]) > 1000:
            print('L diverges weights too large')
            break
    
    # Calculate the Mutual Information
    Hxx, Hxy, MI = MI_est(L, x)
    return [Hxx, Hxy, MI, L]
