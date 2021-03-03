'''
    balance_compare.py

    Compare the Mutual information of different Excitation Inhibition balances.
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

# Set parameters
baseline = 0  
amplitude_scaling = 10
dynamic_scaling = 8
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
N_runs = 10 # for all pyramidal and interneuron parameters


# Generate 
## Input, Hiddenstate and Model
print('Generating...')
[exc_LUT, inh_LUT, input_theory, hidden_state] = make_dynamic_experiments(qon_qoff_type, baseline, amplitude_scaling, tau, factor_ron_roff, mean_firing_rate, sampling_rate, duration, dv)
dynamic_neuron = Barrel_PC('dynamic', dt)
dynamic_neuron.store()
current_neuron = Barrel_PC('current', dt)
current_neuron.store()
print('Input and hiddenstate generate!')

current_input = scale_input_theory(input_theory, baseline, amplitude_scaling, dt)

scale_exc_inh = [15, 20, 25, 30, 35, 40]
Output = {}
FI = []
for scale in scale_exc_inh:
    print('Testing scale',scale)
    dynamic_neuron.restore()
    current_neuron.restore()
    # Scale
    dynamic_input = scale_dynamic_input(exc_LUT, inh_LUT, Er_exc, Er_inh, scale, dt)

    # Simulate
    dynamic_M, dynamic_S = dynamic_neuron.run(dynamic_input, duration*ms, Ni=1)
    current_M, current_S = current_neuron.run(current_input, duration*ms, Ni=1)

    # Sanity Check
    plot_dynamicclamp(dynamic_M, dynamic_input[0], dynamic_input[1], hidden_state, dt=dt)
    plot_currentclamp(current_M, hidden_state, dt)

    # Output[scale] = analyze_exp(ron, roff, hidden_state, input_theory, dt, theta, spiketrain)
    # fi = Output[scale]['MI'] / Output[scale]['MI_i']
    # FI.append(fi)
    # print('scale=',scale, 'MI=',fi)

# plt.plot(scale_exc_inh, FI)
# plt.show()

