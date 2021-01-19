'''
    dynamic_clamp.py

    Generate an input based on a vector of conductances and more.
'''
import numpy as np

#not for permanent use
sampling_rate = 5      
dt = 1/sampling_rate 
T = 20000

weights = [[0.05], [0.04], [0.05], [0.326], [-0.551], [1.25], [1.589], [-0.554]]
weights = np.array(weights)

time_vec = np.arange(dt, T+dt, dt)
volt_vec = np.arange(-100, 75, 5)

def get_conductance(volt_vec, v_rest, weights):
    '''docstring
    '''
    N = len(weights)
    weight_matrix = {}

    for i in range(N):
        weight_matrix[i] = {}
        w = weights[i]

        #Get conductance value based on g(-v_rest - Ei) = w at Vm = -65
        if w > 0:
            Er = 0
            g = w / (-v_rest - Er)
        else:
            Er = -90
            g = w / (-v_rest - Er)
 
        #Get weight for all the voltages
        for v in volt_vec:
            weight_matrix[i][v] = float(g * (-v - Er)) 

    return weight_matrix

test = get_conductance(volt_vec, -65, weights)
print(test[0][-65], weights[0])