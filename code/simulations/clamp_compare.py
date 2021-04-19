''' clamp_compare.py

    Simulation that compares the Mutual Information of Pyramidal cells and Interneurons between
    current clamp and dynamic clamp setup.
'''

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np
from brian2 import *
from foundations.make_dynamic_experiments import make_dynamic_experiments
from models.models import *
from visualization.plotter import plot_currentclamp, plot_dynamicclamp, plot_compare
from foundations.MI_calculation import analyze_exp
from foundations.helpers import scale_input_theory, scale_to_freq, make_spiketrain
from models.models import Barrel_PC
import seaborn as sns

## Generate input and hiddenstate
# Set parameters
baseline = 0  
theta = 0     
tau = 50               
amplitude_scaling = 0
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
on_off_ratio = 1.5

## Simulate
# Pyramidal cells
MI = {'PC_current' : [], 'PC_dynamic' : [], 'IN_current' : [], 'IN_dynamic' : []}
print('Running simulation...')

current_barrel_PC = Barrel_PC('current', dt=dt)
dynamic_barrel_PC = Barrel_PC('dynamic', dt=dt)
current_barrel_PC.store()
dynamic_barrel_PC.store()

current_barrel_IN = Barrel_IN('current', dt=dt)
dynamic_barrel_IN = Barrel_PC('dynamic', dt=dt)
current_barrel_IN.store()
dynamic_barrel_IN.store()

for _ in range(25):
    # Generate input
    print('Generating...')
    [g_exc, g_inh, input_theory, hidden_state] = make_dynamic_experiments(qon_qoff_type, baseline, amplitude_scaling, tau, factor_ron_roff, mean_firing_rate, sampling_rate, duration, dv)
    print('Input and hiddenstate generate!')
    for i in range(1, 12): #N_runs[0]):
        print('Run', i)
        current_barrel_PC.restore()
        dynamic_barrel_PC.restore()

        ## Create and scale input to a Brian2 TimedArray
        # Currentclamp
        input_current = scale_to_freq(current_barrel_PC, input_theory, target, on_off_ratio, 'current', duration, hidden_state, dt, i)

        # Dynamicclamp
        input_dynamic = scale_to_freq(dynamic_barrel_PC, (g_exc, g_inh), target, on_off_ratio, 'dynamic', duration, hidden_state, dt, i)

        # Check if scale was correct
        if input_dynamic != False and input_current != False:
            # Run simulation
            M_current, S_current = current_barrel_PC.run(input_current, duration*ms, Ni=i)
            M_dynamic, S_dynamic = dynamic_barrel_PC.run(input_dynamic, duration*ms, Ni=i)

            # Create spiketrain
            spiketrain_current = make_spiketrain(S_current, hidden_state, dt)
            spiketrain_dynamic = make_spiketrain(S_dynamic, hidden_state, dt)

            # Calculate MI
            Output_current = analyze_exp(ron, roff, hidden_state, input_theory, dt, theta, spiketrain_current)
            Output_dynamic = analyze_exp(ron, roff, hidden_state, input_theory, dt, theta, spiketrain_dynamic)
            MI['PC_current'].append(Output_current)
            MI['PC_dynamic'].append(Output_dynamic)
            
            # # Sanity check
            # print(Output_dynamic['MI'])
            # plot_dynamicclamp(M_dynamic, input_dynamic[0], input_dynamic[1], hidden_state, dt=dt)
            # print(Output_current['MI'])
            # plot_currentclamp(M_current, hidden_state, dt=dt)

# Plot
# print(MI['PC_current'])
# MI_PC_current = [run['MI'] for run in MI['PC_current']]
# MI_PC_dynamic = [run['MI'] for run in MI['PC_dynamic']]
# x = np.arange(2)
# fig, ax = plt.subplots()
# bar = sns.barplot(data=[MI_PC_current, MI_PC_dynamic])
# ax.set_xticklabels(['current', 'dynamic'])
# plt.show()

    for i in range(8, 12):
        # Clamps
        current_barrel_IN.restore()
        dynamic_barrel_IN.restore()
        
        ## Create and scale input to a Brian2 TimedArray
        # Currentclamp
        input_current = scale_to_freq(current_barrel_IN, input_theory, target, on_off_ratio, 'current', duration, hidden_state, dt, i)

        # Dynamicclamp
        input_dynamic = scale_to_freq(dynamic_barrel_IN, (g_exc, g_inh), target, on_off_ratio, 'dynamic', duration, hidden_state, dt, i)
        
        # Check if scale was correct
        if input_dynamic != False and input_current != False:
            # Run simulation
            M_current, S_current = current_barrel_PC.run(input_current, duration*ms, Ni=i)
            M_dynamic, S_dynamic = dynamic_barrel_PC.run(input_dynamic, duration*ms, Ni=i)

            # Create spiketrain
            spiketrain_current = make_spiketrain(S_current, hidden_state, dt)
            spiketrain_dynamic = make_spiketrain(S_dynamic, hidden_state, dt)

            # Calculate MI
            Output_current = analyze_exp(ron, roff, hidden_state, input_theory, dt, theta, spiketrain_current)
            Output_dynamic = analyze_exp(ron, roff, hidden_state, input_theory, dt, theta, spiketrain_dynamic)
            MI['IN_current'].append(Output_current)
            MI['IN_dynamic'].append(Output_dynamic)
            

#     # # Sanity check
#     # print(Output_dynamic['MI'])
#     # plot_dynamicclamp(M_dynamic, g_exc, g_inh, hidden_state, dt=dt)
#     # print(Output_current['MI'])
#     # plot_currentclamp(M_current, hidden_state, dt=dt)

print('Simulation complete, saving files')

# Save files
np.savetxt(f'results/saved/clamp_compare2/hiddenstate.csv', hidden_state, delimiter=',')
np.savetxt(f'results/saved/clamp_compare2/input_theory.csv', input_theory, delimiter=',')
np.savetxt(f'results/saved/clamp_compare2/spiketrain_current.csv', spiketrain_current, delimiter=',')
np.savetxt(f'results/saved/clamp_compare2/spiketrain_dynamic.csv', spiketrain_dynamic, delimiter=',')
np.save(f'results/saved/clamp_compare2/MI.npy', MI)



