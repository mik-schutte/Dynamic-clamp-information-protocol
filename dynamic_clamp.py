'''
    dynamic_clamp.py

    Generate an input based on a vector of conductances and more.
'''
import numpy as np
from matplotlib import pyplot as plt

#not for permanent use
sampling_rate = 5.      
dt = round(1/sampling_rate, 4) 
T = 20000

weights = [[0.5], [0.4], [0.05], [0.326], [-0.551], [1.25], [1.589], [-0.554]]
weights = np.array(weights)

time_vec = np.arange(0, T+dt, dt).round(2)
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
    #TODO extremely low conductance values
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
    h = abs(time_vec[0] - time_vec[1])
    
    A = np.sqrt(D * tau / 2 * (1 - np.exp(-2 * h / tau) ))

    condar = {}
    condar[0] = g0
    for t in time_vec:
        th = round(t+h, 3) 
        condar[th] = g0 + (condar[t] - g0) * np.exp(-h/tau) + A * np.random.normal()
    return condar 


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

#Test
g0 = 0.01

tau = 2.7
sigma = 0.0030
D = 2 * sigma**2 / tau
h = 0.2
A = np.sqrt(D*tau/2 * (1-np.exp(-2*h/tau)))

condar = {}
condar[0] = g0
for t in time_vec:
    tdt = round(t+dt, 3) 
    condar[tdt] = g0 + (condar[t] - g0) * np.exp(-dt/tau) + A * np.random.normal()

# plt.plot(condar.keys(), condar.values(), lw=1)
# plt.show()
x = np.random.normal()
print(time_vec[1] - time_vec[0])