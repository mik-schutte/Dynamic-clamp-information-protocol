'''
    dynamic_clamp.py

    Generate an input based on a vector of conductances and more.
'''
import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt

#not for permanent use
sampling_rate = 5      
dt = 1/sampling_rate 
T = 20000

weights = [[0.05], [0.04], [0.05], [0.326], [-0.551], [1.25], [1.589], [-0.554]]
weights = np.array(weights)

time_vec = np.arange(dt, T+dt, dt).round(2)
volt_vec = np.arange(-100, 75, 5)

def get_g0(v_rest, weights):
    ''' Creates a dictionary containing the mean conductance of each neuron
        in the ANN. Neuron index is used as dictionary key.

        INPUT:
              v_rest (int): resting membrane potential of the neurons in mV.
              weights (array): weights of each ANN neuron.
              time_vec (array): vector of time with dt timesteps. 

        The reversal potential (Er) is based on A. Destexhe, M. Rudolph, J.M. Fellous 
        & T.J. Sejnowski (2001).
    '''
    #TODO are there really negative conductances?
    #TODO If g negative but weight positive?
    #TODO np.random.normal should be replaced by the Destexhe (2001) equation
    N = len(weights)

    #Generate vector of conductances
    g_dict = {}
    for i in range(N):
        if weights[i] > 0:
            Er = 0
            g0 = float(weights[i] / (-v_rest - Er))
            
        else: 
            Er = -75
            g0 = float(weights[i] / (-v_rest - Er))
        g_dict[i] = g0

    return g_dict


def differential_conductance(g, t, g0, tau, D, X):
    '''docstring
    '''
    print('t=', t)
    dgdt = -1/tau * (g - g0) + np.sqrt(D) * X(t)
    return dgdt


def get_point_conductance(g0, tau, sigma, time_vec):
    ''' Generate g(t) 

        INPUT:
              g0 (dict): 
              tau (float):
              sigma (float): standard deviation
              time_vec :  
    '''
    #Calculate noise 'diffusion' coefficient.
    D = 2 * sigma**2 / tau
    
    #Calculate Gaussian white noise with mean=0 and sd=sigma for all time points
    X = np.random.normal(loc=0, scale=sigma, size=len(time_vec))

    #Integrate 
    #Initial condition
    g_start = 0 #np.zeros(X.shape)
    print(g_start)
    g = odeint(differential_conductance, g_start, time_vec, args=(g0, tau, D, X,))
    return g


def get_input_LUT(volt_vec, g_dict):
    ''' Create a look-up table (LUT) of injected currents based on conductance(t) of
        each ANN neuron and the theoretical voltage of the receiving neuron.

        INPUT:
              volt_vec: vector of theoretical voltage values with dv.
              g_dict: dictionary of individual conductances over time.
    '''
    #TODO get all keys from g_dict in stead of for i in range(N)
    #TODO if 0 volt input is also 0
    #TODO we don't really have a volt for the ANN neurons.

    N = len(g_dict)
    input_dict = {}

    for i in range(N):
        input_dict[i] = {}
        for v in volt_vec:
            input_dict[i][v] = v * g_dict[i]

    return input_dict

g_dict = get_g0(-65, weights)
#input_dict = get_input_LUT(volt_vec, g_dict)

#Test
g0 = g_dict[0]
tau = 2.7
sigma = 0.0030
D = 2 * sigma**2 / tau
X = np.random.normal(loc=0, scale=sigma, size=len(time_vec))

test = get_point_conductance(g0, tau, sigma, time_vec)

