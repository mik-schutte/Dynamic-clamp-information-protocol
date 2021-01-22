'''
    make_dynamic_experiments.py

'''
import numpy as np
import matplotlib.pyplot as plt
from dynamic_clamp import * 

def make_dynamic_experiments(weights, tau, sigma, v_rest, volt_vec, time_vec):
    '''docstring
    '''
    assert isinstance(time_vec, (list, np.ndarray)), 'time_vec must be a list or numpy.ndarray'
    if isinstance(time_vec, list):
        time_vec = np.array(time_vec)
    time_vec = time_vec.round(3)

    assert isinstance(weights, (list, np.ndarray)), 'weights must be a list or numpy.ndarray'
    if type(weights) == list:
        weights = np.array(weights)

    #Seperate inhibitory form excitatory neurons
    w_e = weights[weights>0]
    w_i = weights[weights<=0]

    #Get g0 and stochastic conductance
    g0_e_dict = get_g0(v_rest, w_e)
    g0_i_dict = get_g0(v_rest, w_i)
    g_e = get_stochastic_conductance(g0_e_dict, tau, sigma, time_vec)
    g_i = get_stochastic_conductance(g0_i_dict, tau, sigma, time_vec) 

    #Generate input LUT
    inputLUT_e = get_input_LUT(g_e, volt_vec, 0)
    inputLUT_i = get_input_LUT(g_i, volt_vec, -75)  #Can now be different from g0's Er

    return inputLUT_e, inputLUT_i

#Test 
weights = np.array([.5, 0.4, 0.8, 1.2, 0.08, -0.5, -1])
tau = 2.7
sigma = 0.0030
v_rest = -65
time_vec = np.arange(0, 2000, 0.2).round(3)
volt_vec = np.arange(-100, 25, 5)

teste, testi = make_dynamic_experiments(weights, tau, sigma, v_rest, volt_vec, time_vec)

