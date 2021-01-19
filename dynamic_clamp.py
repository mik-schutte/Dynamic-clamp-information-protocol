'''
    dynamic_clamp.py

    Generate an input based on a vector of conductances and more.
'''
import numpy as np
from matplotlib import pyplot as plt

#not for permanent use
sampling_rate = 5      
dt = 1/sampling_rate 
T = 20000

weights = [[0.05], [0.04], [0.05], [0.326], [-0.551], [1.25], [1.589], [-0.554]]
weights = np.array(weights)

time_vec = np.arange(dt, T+dt, dt)
volt_vec = np.arange(-100, 75, 5)

def get_conductance(v_rest, weights, time_vec):
    ''' docstring
    sd based on A. Destexhe, M. Rudolph, J.M. Fellous & T.J. Sejnowski (2001)
    '''
    N = len(weights)

    #Generate vector of conductances
    g_dict = {}
    for i in range(N):
        if weights[i] > 0:
            Er = 0
            g0 = float(weights[i] / (-v_rest - Er))
            sd = 0.0014
        else: 
            Er = -90
            g0 = float(weights[i] / (-v_rest - Er))
            sd = 0.0029

        # Generate fluctuating conductance based on g0 and sd 
        g_dict[i] = np.random.normal(loc=g0, scale=sd, size=len(time_vec))

    #TODO are there really negative conductances?
    #TODO If g negative but weight positive?
    #TODO np.random.normal should be replaced by the Destexhe (2001) equation
    return g_dict


def get_input(volt_vec, g_dict):
    '''docstring
    '''
    N = len(g_dict)
    input_dict = {}

    for i in range(N):
        input_dict[i] = {}
        for v in volt_vec:
            input_dict[i][v] = v * g_dict[i]

    #TODO get all keys from g_dict in stead of for i in range(N)
    #TODO if 0 volt input is also 0
    #TODO we don't really have a volt for the ANN neurons.
    return input_dict

g_dict = get_conductance(-65, weights, time_vec)
input_dict = get_input(volt_vec, g_dict)
print(input_dict[0])