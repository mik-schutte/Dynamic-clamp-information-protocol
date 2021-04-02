''' ISI_compare.py

    Simulation that calculates the inter spike interval (ISI) of Pyramidal cells and Interneurons
    in response to the hidden state in both current clamp and dynamic clamp setup.
'''
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from foundations.make_dynamic_experiments import make_dynamic_experiments
from models.models import *
from foundations.helpers import *
import numpy as np
import matplotlib.pyplot as plt
from visualization.plotter import plot_currentclamp, plot_dynamicclamp

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
target = 12
N_runs = (61, 22) # for all pyramidal and interneuron parameters

ISI = {'current_PC':[], 'dynamic_PC':[], 'current_IN':[], 'dynamic_IN':[]}
current_PC = Barrel_PC('current', dt)
dynamic_PC = Barrel_PC('dynamic', dt)
current_IN = Barrel_IN('current', dt)
dynamic_IN = Barrel_IN('dynamic', dt)
current_PC.store()
dynamic_PC.store()
current_IN.store()
dynamic_IN.store()

print('Running simulation') 
for _ in range(10):
    # Generate input
    [g_exc, g_inh, input_theory, hidden_state] = make_dynamic_experiments(qon_qoff_type, baseline, amplitude_scaling, tau, factor_ron_roff, mean_firing_rate, sampling_rate, duration, dv)
    
    # Pyramidal Cells
    for i in range(N_runs[0]):
        print(f'Simulating PC {i+1} of {N_runs[0]}.')
        current_PC.restore()
        dynamic_PC.restore()
        
        # Scale input
        current_scale = scale_to_freq(current_PC, input_theory, target, 'current', duration, dt, Ni=i)
        dynamic_scale = scale_to_freq(dynamic_PC, (g_exc, g_inh), target, 'dynamic', duration, dt, Ni=i)

        inj_current = scale_input_theory(input_theory, baseline, current_scale, dt)
        inj_dynamic = scale_dynamic_input(g_exc, g_inh, dynamic_scale, dt)

        # Simulate and calculate
        current_M, current_S = current_PC.run(inj_current, duration*ms, Ni=i)
        dynamic_M, dynamic_S = dynamic_PC.run(inj_dynamic, duration*ms, Ni=i)
        ISI['current_PC'] = np.concatenate((ISI['current_PC'], get_spike_intervals(current_S)), axis=0)
        ISI['dynamic_PC'] = np.concatenate((ISI['dynamic_PC'], get_spike_intervals(dynamic_S)), axis=0)

        # Sanity Check:
        # plot_currentclamp(current_M, hidden_state, dt)
        # plot_dynamicclamp(dynamic_M, inj_dynamic[0], inj_dynamic[1], hidden_state, dt)

    # Interneurons
    for i in range(N_runs[1]):
        # print(f'Simulating IN {i} of {N_runs[1]}.')
        current_IN.restore()
        dynamic_IN.restore()

        # Scale input
        current_scale = scale_to_freq(current_IN, input_theory, target, 'current', duration, dt, Ni=i)
        dynamic_scale = scale_to_freq(dynamic_IN, (g_exc, g_inh), target, 'dynamic', duration, dt, Ni=i)

        inj_current = scale_input_theory(input_theory, baseline, current_scale, dt)
        inj_dynamic = scale_dynamic_input(g_exc, g_inh, dynamic_scale, dt)

        # Simulate and calculate
        current_M, current_S = current_IN.run(inj_current, duration*ms, Ni=i)
        dynamic_M, dynamic_S = dynamic_IN.run(inj_dynamic, duration*ms, Ni=i)
        ISI['current_IN'] = np.concatenate((ISI['current_IN'], get_spike_intervals(current_S)), axis=0)
        ISI['dynamic_IN'] = np.concatenate((ISI['dynamic_IN'], get_spike_intervals(dynamic_S)), axis=0)

        # Sanity Check:
        # plot_currentclamp(current_M, hidden_state, dt)
        # plot_dynamicclamp(dynamic_M, inj_dynamic[0], inj_dynamic[1], hidden_state, dt)

# # Save ISI dictionary
np.save(f'results/saved/ISI_compare/ISI.npy', ISI)

# Plot a lot
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
axs[0, 0].hist(ISI['current_PC'], bins=100, label='current PC')
axs[0, 1].hist(ISI['dynamic_PC'], bins=100, label='dynamic PC')
axs[1, 0].hist(ISI['current_IN'], bins=100, label='current IN')
axs[1, 1].hist(ISI['dynamic_IN'], bins=100, label='dynamic IN')
axs[0, 0].title.set_text('current PC')
axs[0, 1].title.set_text('dynamic PC')
axs[1, 0].title.set_text('current IN')
axs[1, 1].title.set_text('dynamic IN')
plt.legend()
plt.show()
