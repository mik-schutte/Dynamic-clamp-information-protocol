'''
    docsting
'''

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np
from foundations.make_dynamic_experiments import make_dynamic_experiments
from foundations.MI_calculation import analyze_exp
from visualization.plotter import plot_dynamicclamp, plot_currentclamp
from models.models import Barrel_PC
from brian2 import *
from foundations.helpers import scale_dynamic_input, make_spiketrain, scale_input_theory
from visualization.plotter import plot_scaling_compare

# Set parameters
baseline = 0  
amplitude_scaling = 7.5
dynamic_scaling = 7.5
theta = 0     
tau = 50               
factor_ron_roff = 2    
ron = 1./(tau*(1+factor_ron_roff))
roff = factor_ron_roff*ron
mean_firing_rate = (0.5)/1000 
sampling_rate = 2      
dt = 1/sampling_rate #0.5 ms so that the barrel models work
dv = 0.5
duration = 2000
qon_qoff_type = 'balanced'
Er_exc, Er_inh = (0, -75)
N_runs = 5 # for all pyramidal and interneuron parameters

Er_inh_array = [-75, -80, -90, -100, -200, -300]
scaled_inputs = dict.fromkeys(Er_inh_array, [])
scaled_Vm = dict.fromkeys(Er_inh_array, [])
scaled_freq = dict.fromkeys(Er_inh_array, [])
scaled_freqdiff = dict.fromkeys(Er_inh_array, [])
current_inputs = []         
current_Vm = []   
current_freq = []

for i in range(N_runs):
    # Generate 
    ## Input, Hiddenstate and Model
    print('Generating...')
    [g_exc, g_inh, input_theory, hidden_state] = make_dynamic_experiments(qon_qoff_type, baseline, amplitude_scaling, tau, factor_ron_roff, mean_firing_rate, sampling_rate, duration, dv)
    print('Input and hiddenstate generate!')

    # Current Clamp 
    current_neuron = Barrel_PC('current', dt)
    current_inj = scale_input_theory(input_theory, baseline, amplitude_scaling, dt)
    current_M, current_S = current_neuron.run(current_inj, duration*ms, 1, Er_exc, Er_inh)
    current_inputs = np.concatenate((current_inputs, current_M.I_inj[0]/uA), axis=0)
    current_Vm = np.concatenate((current_Vm, current_M.v[0]/mV), axis=0)
    current_freq = np.concatenate((current_freq, [current_S.num_spikes/(duration/1000)]), axis=0)

    # Dynamic Clamp
    dynamic_neuron = Barrel_PC('dynamic', dt)
    dynamic_neuron.store()

    # Dynamic Clamp with different Er_inh
    for Er_inh in Er_inh_array:
        print('Testing Er', Er_inh)
        dynamic_neuron.restore()

        # Scale
        dynamic_input = scale_dynamic_input(g_exc, g_inh, dynamic_scaling, dt)

        # Simulate
        dynamic_M, dynamic_S = dynamic_neuron.run(dynamic_input, duration*ms, 1, Er_exc, Er_inh)

        # Sanity Check
        # if i == 0:
        #     plot_dynamicclamp(dynamic_M, dynamic_input[0], dynamic_input[1], hidden_state, dt=dt)
        #     plot_currentclamp(current_M, hidden_state, dt)
        
        scaled_inputs[Er_inh] = np.concatenate((scaled_inputs[Er_inh], dynamic_M.I_inj[0]/uA), axis=0)
        scaled_Vm[Er_inh] = np.concatenate((scaled_Vm[Er_inh], dynamic_M.v[0]/mV), axis=0)
        scaled_freq[Er_inh] = np.concatenate((scaled_freq[Er_inh], [dynamic_S.num_spikes/(duration/1000)]), axis=0) 
        scaled_freqdiff[Er_inh] = np.concatenate((scaled_freqdiff[Er_inh], [scaled_freq[Er_inh][i] - current_freq[i]]), axis=0) 

current_dict = {'I':current_inputs, 'Vm':current_Vm, 'f':current_freq}
dynamic_dict = {'I':scaled_inputs, 'Vm':scaled_Vm, 'f':scaled_freq, 'fdiff': scaled_freqdiff}

# Plot
plot_scaling_compare([current_dict, dynamic_dict])

# Save
np.save('results/saved/Er_compare/current_dict.npy', current_dict)
np.save('results/saved/Er_compare/dynamic_dict.npy', dynamic_dict)
