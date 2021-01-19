'''
    dynamic_clamp.py

    Generate an input based on a vector of conductances and more.
'''
import numpy as np
import matplotlib.pyplot as plt  

#not for permanent use
sampling_rate = 5      
dt = 1/sampling_rate 
T = 20000
weights = [[0.05], [0.04], [0.05], [0.326], [-0.551], [1.25], [1.589], [-0.554]]
weights = np.array(weights)
#Inhibitory and Excitatory reversal
E_e = 0
E_i = -90

time_vec = np.arange(dt, T+dt, dt)
volt_vec = np.arange(-100, 60.5, 0.5)
g = np.sin(time_vec * np.pi/180) + 0.05


#Test sinus
def get_conductance(volt_vec, v_rest, weights):
    ''' docstring
    '''
    #Inhibitory and Excitatory reversal
    E_e = 0
    E_i = -90

    #Generate vector of conductances
    g_vec = []
    for w in weights:
        if w > 0:
            g = w / (-v_rest - E_e)
            new_w = g * (-v - E_e)
        else: 
            g = w / (-v_rest - E_i)

        g_vec.append(g)

    #TODO are there really negative conductances?
    return np.array(g_vec).T    


test = get_conductance(-65, weights)
