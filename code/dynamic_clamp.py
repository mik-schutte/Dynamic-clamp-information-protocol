'''
    dynamic_clamp.py

    File containing the functions used in make_dynamic_experiments and main(dynamic).
'''
import numpy as np
import code.input as Input

def get_g0(v_rest, weights, Er_exc, Er_inh):
    ''' Creates a dictionary containing the 'base' conductance of each neuron
        in the ANN. Neuron index is used as dictionary key. 

        INPUT:
              v_rest(int): resting membrane potential of the neurons in mV.
              weights(array): weights of each ANN neuron.
              Er_exc/inh(int): reversal potential of exc. and inh. neurons in mV.
        OUTPUT: 
              [g0_exc_dict, g0_inh_dict]: dictionaries of 'base' conductances with 
                                          neuron index as key.

        Base conductance: value where g0 * (Vrest - Er) = weight.
        The reversal potential (Er) is based on A. Destexhe, M. Rudolph, J.M. Fellous 
        & T.J. Sejnowski (2001). 
    '''
    N = len(weights)

    #Initiate dictionaries
    g0_exc_dict = {}
    g0_inh_dict = {}

    #Get g0 and seperate in to inhibitory and excitatory conductance
    for i in range(N):
        if weights[i] > 0:
            g0 = float(weights[i] / (-v_rest - Er_exc))
            g0_exc_dict[i] = abs(g0)
        else: 
            g0 = float(weights[i] / (-v_rest - Er_inh))
            g0_inh_dict[i] = abs(g0)

    return [g0_exc_dict, g0_inh_dict]


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
    N = len(g0_dict)
    D = 2 * sigma**2 / tau                                  #Noise 'diffusion' coefficient
    A = np.sqrt(D * tau / 2 * (1 - np.exp(-2 * dt / tau) )) #Amplitude coefficient
    
    #Initiate dictionary
    sto_cond = {}
    for i in range(N):
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


def get_input_LUT(sto_cond, dv, Er):
    ''' Create a look-up table (LUT) of injected currents based on conductance(t) 
        and the voltages from -100 to +20 mV.

        INPUT:
              sto_cond(array): dictionary of individual conductances over time.
              dv(int): resolution of the voltage steps minimal 0.001.
              Er(int): inhibitory or excitatory conductances?
        OUTPUT:
              input_LUT(dict): keys are the voltage and value the I(t). 
    '''  
    volt_vec = np.arange(-100, 20+dv, dv).round(3)
    input_LUT = {}
    for v in volt_vec:
        input_LUT[v] = sto_cond * (-v - Er)

    return input_LUT
