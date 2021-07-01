''' MI_calculation.py
    File containing the functions to estimate the mutual information betweem the hidden state, input and 
    output spike train. 

    Described in:
    Zeldenrust, F., de Knecht, S., Wadman, W. J., Denève, S., Gutkin, B., Knecht, S. De, Denève, S. (2017). 
    Estimating the Information Extracted by a Single Spiking Neuron from a Continuous Input Time Series. 
    Frontiers in Computational Neuroscience, 11(June), 49. doi:10.3389/FNCOM.2017.00049
    Please cite this reference when using this method.
'''
import numpy as np
import pandas as pd
from scipy import stats, integrate

def analyze_exp(ron, roff, x, input_theory, dt, theta, spiketrain): 
    ''' Analyzes the the hidden state and the input that was created by the ANN to
        create the Output dictionary.
        Equations 13 & 14

        INPUT:
        ron, roff (kHz): switching speed of the hidden state
        x: array with hidden state values over time
        input_theory: array with unscaled input current values (output from ANN)
        dt: binsize recordings (ms)
        spiketrain: array (same size as x and input_theory) of 0 (no spike) and 1 (spike)

        OUTPUT
        Output-dictionary with keys:
        MI_i        : mutual information between hidden state and input current
        xhat_i      : array with hidden state estimate based on input current 
        MSE_i       : mean-squared error between hidden state and hidden state estimate based on input current 
        MI          : mutual information between hidden state and spike train
        qon, qoff   : spike frequency during on, off state in spike train 
        xhatspikes  : array with hidden state estimate based on spike train
        MI          : mean-squared error between hidden state and hidden state estimate based on spike train
    '''
    Output = {}
    # Input
    Hxx, Hxy, Output['MI_i'], L_i = calc_MI_input(ron, roff, input_theory, theta, x , dt)
    Output['xhat_i'] = 1. / (1 + np.exp(-L_i))
    Output['MSE_i'] = np.sum((x - Output['xhat_i'])**2)

    # Output
    _, _, Output['MI'], L, Output['qon'], Output['qoff'] = calc_MI_ideal(ron, roff, spiketrain, x, dt)
    Output['xhatspikes'] = 1./(1 + np.exp(-L))
    Output['MSE'] = np.sum((x - Output['xhatspikes'])**2)

    return pd.DataFrame.from_dict(Output, orient='index').T


def dLdt_input(L, ron, roff, I, theta):
    ''' Differential equation calculating the posterior Log-likelihood of the 
        hidden state being 1 based on the input history.
        Equation 10.
    '''        
    dLdt = ron * (1. + np.exp(-L)) - roff * (1. + np.exp(L)) + I - theta

    return dLdt


def dLdt_spikes(L, ron, roff, I, w, theta):
    'docstring'
    dLdt = ron * (1. + np.exp(-L)) - roff * (1. + np.exp(L)) + w*I - theta

    return dLdt


def p_conditional(L):  
    ''' Estimates the probability that the hidden state equals 1 given the input history.
        Equation 11.
    '''
    return 1. / (1 + np.exp(-L))


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


def calc_MI_ideal(ron, roff, spiketrain, x, dt):
    ''' Calculate the (conditional) entropy, MI, and likelihood.
    '''
    ## Calculate qon, qoff, w and theta
    spikesup, spikesdown = reorder_x(x, spiketrain)
    spikesup = np.squeeze(spikesup)
    spikesdown = np.squeeze(spikesdown)

    nspikesup = abs(np.nansum(np.nansum(spikesup)))
    nspikesdown = abs(np.nansum(np.nansum(spikesdown)))
    if nspikesdown == 0:
        print('no down spikes, inventing one')
        nspikesdown = 1 

    qon = nspikesup / (sum(x)*dt)
    qoff = nspikesdown / ((len(x) - sum(x))*dt)
    w = np.log(qon/qoff)
    theta = qon-qoff
    # print('w=', w, '; theta=', theta)

    ## Integrate L
    I = spiketrain/dt
    L = np.empty(np.shape(x))
    L[0] = np.log(ron/roff)

    for nn in range(len(x) - 1):
        L[nn+1] = L[nn] + dLdt_spikes(L[nn], ron, roff, I[0][nn], w, theta) * dt
        if abs(L[nn+1]) > 1000:
            assert StopIteration('L diverges, weights too large')
    
    # Calculate MI
    Hxx, Hxy, MI = MI_est(L, x)

    return Hxx, Hxy, MI, L, qon, qoff


def reorder_x(x, ordervecs):
    ''' Reorder the vectors in ordervec (nvec * length) to x=1 (up) 
        and x=0 (down)
    '''
    ## Check if transposing is necessary
    # Check ordervecs
    number_of_vectors, timesteps = np.shape(ordervecs)
    if number_of_vectors > timesteps:
        s = input('Number of vectors larger than number of timesteps; transpose? (y/n')
        if s == 'y':
            ordervecs = np.transpose(ordervecs)
            number_of_vectors, timesteps = np.shape(ordervecs)
    
    # Check hiddenstate
    number_of_xvectors, _ = np.shape([x])
    if number_of_xvectors != 1:
        x = np.transpose(x)
    
    ## TODO what does this piece of code do?
    xt1 = np.insert(x, 0, x[0])
    xt2 = np.append(x, x[-1])
    xj = xt2 - xt1 # This is equal to xj[n] = x[n] - x[n-1]
    njumpup = len(xj[xj==1])
    njumpdown = len(xj[xj==-1])

    ## Reorder
    if njumpup > 0 and njumpdown > 0:
        firstjump = np.argwhere(abs(xj) == 1).flatten()
        firstjump = firstjump[0]
        revecsup = np.nan * np.empty((number_of_vectors, njumpup+1, round(10*timesteps/njumpup)))
        revecsdown = np.nan * np.empty((number_of_vectors, njumpdown+1, round(10*timesteps/njumpdown)))
        _, _, size3 = np.shape(revecsdown)

        if x[firstjump] == 1:
            up = 1
            down = 0
            revecsup[:, 0, 0] = ordervecs[:, firstjump]
        elif x[firstjump] == 0:
            up = 0
            down = 1
            revecsdown[:, 0, 0] = ordervecs[:, firstjump]
        else:
            raise AssertionError('First jump is not properly defined')

        tt = 1
        tmaxup = 1
        tmaxdown = 1

        for nn in range(firstjump+1, timesteps):
            try:
                jump = x[nn] - x[nn-1]
            except:
                raise AssertionError('size ordervecs not the same as size x')
            
            # Make jumps
            if jump == 0:
                tt = tt + 1
                
                if x[nn] == 1:
                    # Up state
                    if tt > tmaxup:
                        tmaxup = tt
                    revecsup[:, up, tt] = ordervecs[:, nn]
                elif x[nn] == 0:
                    # Down state
                    if tt > tmaxdown:
                        tmaxdown = tt
                    revecsdown[:, down, tt] = ordervecs[:, nn]
                else:
                    raise AssertionError('Something went wrong: x not 0 or 1')
            
            elif jump == 1:
                #Jump up
                tt = 1
                up = up + 1
                if x[nn] == 1:
                    revecsup[:, up, tt] = ordervecs[:, nn]
                else:
                    raise AssertionError('Something went wrong: jump up but x not 1')

            elif jump == -1:
                # Jump down
                tt = 1
                down = down + 1
                if x[nn] == 0:
                    revecsdown[:, down, tt] = ordervecs[:, nn]
                else:
                    raise AssertionError('Something went wrong: jump down but x is not 0')
            
            else:
                raise AssertionError('Something went wrong: no jump up or down')
        
            if tt > size3 - 1:
                # tt will run out of Matrix size
                raise IndexError('Choose larger starting Matrix')

        revecsup = revecsup[:, 1:up, 1:tmaxup] 
        revecsdown = revecsdown[:, 1:down, 1:tmaxdown]
    
    else:
        if njumpup < 1:
            print('No jumps up; reordering not possible')
            revecsup = None
            revecsdown = None
        if njumpdown < 1:
            print('No jumps down; reordering not possible')
            revecsup = None
            revecsdown = None
            
    return revecsup, revecsdown
