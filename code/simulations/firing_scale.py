'''
    firing_scale.py

    Scale the input based on firing rate
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
theta = 0     
tau = 50     
amplitude_scaling = 7.5          
factor_ron_roff = 2    
ron = 1./(tau*(1+factor_ron_roff))
roff = factor_ron_roff*ron
mean_firing_rate = (0.5)/1000 
sampling_rate = 2      
dt = 1/sampling_rate #0.5 ms so that the barrel models work
dv = 0.5
duration = 2000
qon_qoff_type = 'balanced'
Er_exc, Er_inh = (0, -255)

# Generate input and hiddenstate
print('Generating...')
[g_exc, g_inh, input_theory, hidden_state] = make_dynamic_experiments(qon_qoff_type, baseline, amplitude_scaling, tau, factor_ron_roff, mean_firing_rate, sampling_rate, duration, dv)
print('Input and hiddenstate generate!')

current_neuron = Barrel_PC('current', dt)
current_neuron.store()

dynamic_neuron = Barrel_PC('dynamic', dt)
dynamic_neuron.store()

scales = [2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25, 27.5, 30, 32.5, 35]

def firing_scale()
firing_ratemiss = {'current': {}, 'dynamic': {}} #dict.fromkeys(scales, None)
# Go through different scales
for scale in scales:
    current_neuron.restore()
    dynamic_neuron.restore()

    # Scale 
    inj_current = scale_input_theory(input_theory, baseline, scale, dt)
    inj_dynamic = scale_dynamic_input(g_exc, g_inh, scale, dt)

    # Run
    current_M, current_S = current_neuron.run(inj_current, duration*ms, 1, Er_exc, Er_inh)
    dynamic_M, dynamic_S = dynamic_neuron.run(inj_dynamic, duration*ms, 1, Er_exc, Er_inh)
    
    # Get firing rate
    current_rate = current_S.num_spikes/(duration/1000)
    dynamic_rate = dynamic_S.num_spikes/(duration/1000)

    # Sanity Check
    # plot_currentclamp(current_M, hidden_state, dt)
    # plot_dynamicclamp(dynamic_M, inj_dynamic[0], inj_dynamic[1], hidden_state, dt=dt)

    firing_ratemiss['current'][scale] = abs(current_rate - 12)
    firing_ratemiss['dynamic'][scale] = abs(dynamic_rate - 12)

    print(firing_ratemiss)
    

print('min=', min(firing_ratemiss['current'], key=firing_ratemiss['current'].get))
print('min=', min(firing_ratemiss['dynamic'], key=firing_ratemiss['dynamic'].get))
