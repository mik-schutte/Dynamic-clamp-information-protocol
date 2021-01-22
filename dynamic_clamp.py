'''
    dynamic_clamp.py

    Generate an input based on a vector of conductances and more.
'''
import numpy as np
from matplotlib import pyplot as plt

def get_g0(v_rest, weights):
    ''' Creates a dictionary containing the 'base' conductance of each neuron
        in the ANN. Neuron index is used as dictionary key. 

        INPUT:
              v_rest(int): resting membrane potential of the neurons in mV.
              weights(array): weights of each ANN neuron.
              time_vec(array): vector of time with dt timesteps.
        OUTPUT: 
              g0_dict(dict): dictionary of 'base' conductances with neuron index as 
              key.

        Base conductance: value where g0 * (Vrest - Er) = weight.
        The reversal potential (Er) is based on A. Destexhe, M. Rudolph, J.M. Fellous 
        & T.J. Sejnowski (2001). 
    '''
    #TODO are there really negative conductances?
    #TODO If g negative but weight positive?
    #TODO np.random.normal should be replaced by the Destexhe (2001) equation
    #TODO seperate ge from gi?
    N = len(weights)

    #Generate g0 for every neuron (i)
    g0_dict = {}
    for i in range(N):
        if weights[i] > 0:
            Er = 0
        else: 
            Er = -75
        
        g0 = float(weights[i] / (-v_rest - Er))
        g0_dict[i] = g0

    return g0_dict


def get_stochastic_conductance(g0, tau, sigma, time_vec):
    ''' Generate conductance over time as a stochastic process.  

        INPUT:
              g0(dict): base conductance of neurons with index as key.
              tau(float): time constant
              sigma(float): standard deviation of the conductance.
              time_vec(array): evenly spaced array containing each time point.
        OUTPUT:
              sto_cond(dict): dictionary of conductances with time point as key.

        D, A and update rule are based on A. Destexhe, M. Rudolph, J.M. Fellous 
        & T.J. Sejnowski (2001). 
    '''
    #TODO sto_cond as dict or array?
    
    D = 2 * sigma**2 / tau                                 #Noise 'diffusion' coefficient
    h = abs(time_vec[0] - time_vec[1])                     #Integration step
    A = np.sqrt(D * tau / 2 * (1 - np.exp(-2 * h / tau) )) #Amplitude coefficient

    #Initiate dictionary
    sto_cond = {}
    sto_cond[0] = g0

    #Update dict following an exact update rule
    for t in time_vec[:-1]:
        th = round(t + h, 3) 
        sto_cond[th] = g0 + (sto_cond[t] - g0) * np.exp(-h / tau) + A * np.random.normal()

    #Return as np.array
    return np.fromiter(sto_cond.values(), dtype=float)


def get_input_LUT(volt_vec, sto_cond, Er):
    ''' Create a look-up table (LUT) of injected currents based on conductance(t) 
        and the voltages in volt_vec.

        INPUT:
              volt_vec: vector of voltage values to determine I(t) for.
              sto_cond(array): dictionary of individual conductances over time.
              Er(int): inhibitory or excitatory conductances?
        OUTPUT:
              input_LUT(dict): keys are the voltage and value the I(t). 
    '''
    #TODO get all keys from g_dict in stead of for i in range(N)
    #TODO if 0 volt input is also 0
    #TODO we don't really have a volt for the ANN neurons.
    #TODO if cond is negative split to gi?
    N = len(sto_cond)
    input_LUT = {}

    for v in volt_vec:
        input_LUT[v] = sto_cond * (-v - Er)

    return input_LUT

def get_ANN_LUT(g0_dict, time_vec, tau, sigma, Er):
    


#Test 
weights = [0.5, 0.4, 0.8, 1.2, 0.08]
tau = 2.7
sigma = 0.0030
time_vec = np.arange(0, 20, 0.2).round(3)
volt_vec = np.arange(-100, 25, 5)

g0_dict = get_g0(-65, weights)

for g0 in g0_dict.values():
    sto_cond = get_stochastic_conductance(g0, tau, sigma, time_vec)
    
    input_LUT = get_input_LUT(volt_vec, sto_cond, Er=0)
    
plt.plot(time_vec, input_LUT[-20])
plt.show()

