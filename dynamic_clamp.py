'''
    dynamic_clamp.py

    Generate an input based on a vector of conductances and more.
'''
import numpy as np
import code.input as Input

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


def get_stochastic_conductance(g0_dict, tau, sigma, T, dt):
    ''' Generate conductance over time as a stochastic process.  

        INPUT:
              g0_dict(dict): base conductance of neurons with index as key.
              tau(float): time constant
              sigma(float): standard deviation of the conductance.
              T(int): total duration.
              dt(float): time step.
        OUTPUT:
              sto_cond(dict): dictionary of stochasticconductances with index as key.

        D, A and update rule are based on A. Destexhe, M. Rudolph, J.M. Fellous 
        & T.J. Sejnowski (2001). 
    '''
    D = 2 * sigma**2 / tau                                  #Noise 'diffusion' coefficient
    A = np.sqrt(D * tau / 2 * (1 - np.exp(-2 * dt / tau) )) #Amplitude coefficient

    #Initiate dictionary
    sto_cond = {}
    for i in g0_dict.keys():
        g0 = g0_dict[i]
        sto_cond[i] = {}
        sto_cond[i][0] = g0

        #Update dict following an exact update rule
        for t in np.arange(0, T, dt).round(3):
            tdt = round(t + dt, 3)
            sto_cond[i][tdt] = g0 + (sto_cond[i][t] - g0) * np.exp(-dt / tau) + A * np.random.normal()
        
        #Un-nest dict with index as key and a list of conductances
        sto_cond[i] = np.fromiter(sto_cond[i].values(), dtype=float)
    
    return sto_cond 


def get_input_LUT(sto_cond, volt_vec, Er):
    ''' Create a look-up table (LUT) of injected currents based on conductance(t) 
        and the voltages in volt_vec.

        INPUT:
              sto_cond(array): dictionary of individual conductances over time.
              volt_vec(array): vector of voltage values to determine I(t) for.
              Er(int): inhibitory or excitatory conductances?
        OUTPUT:
              input_LUT(dict): keys are the voltage and value the I(t). 
    '''
    #TODO get all keys from g_dict in stead of for i in range(N)
    #TODO we don't really have a volt for the ANN neurons.
    #TODO if cond is negative split to gi?
    N = len(sto_cond)
    input_LUT = {}

    for v in volt_vec:
        input_LUT[v] = np.empty_like(sto_cond[0])
        for i in range(N):
            input_LUT[v] += sto_cond[i] * (-v - Er)

    return input_LUT
