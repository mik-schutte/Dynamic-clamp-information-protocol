''' scaling_compare.py

    Compare the effect of different scales on the input, membrane potential and firing frequency.
'''
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np
import seaborn as sns
from foundations.make_dynamic_experiments import make_dynamic_experiments
from foundations.MI_calculation import analyze_exp
from visualization.plotter import plot_dynamicclamp, plot_currentclamp
from models.models import Barrel_IN, Barrel_PC
from brian2 import *
from foundations.helpers import make_spiketrain, scale_input_theory
from visualization.plotter import plot_scaling_compare

# Set parameters
baseline = 0  
amplitude_scaling = 7.5
dynamic_scaling = 1
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

scale_exc_inh = np.linspace(0, 150, num=24)
current_inputs_PC = dict.fromkeys(scale_exc_inh, [])
current_Vm_PC = dict.fromkeys(scale_exc_inh, [])
current_freq_PC = dict.fromkeys(scale_exc_inh, [])
dynamic_inputs_PC = dict.fromkeys(scale_exc_inh, [])
dynamic_Vm_PC = dict.fromkeys(scale_exc_inh, [])
dynamic_freq_PC = dict.fromkeys(scale_exc_inh, [])
current_inputs_IN = dict.fromkeys(scale_exc_inh, [])
current_Vm_IN= dict.fromkeys(scale_exc_inh, [])
current_freq_IN = dict.fromkeys(scale_exc_inh, [])
dynamic_inputs_IN = dict.fromkeys(scale_exc_inh, [])
dynamic_Vm_IN = dict.fromkeys(scale_exc_inh, [])
dynamic_freq_IN = dict.fromkeys(scale_exc_inh, [])

# Generate 
## Input, Hiddenstate and Model
[input_theory, dynamic_theory, hidden_state] = make_dynamic_experiments(qon_qoff_type, baseline, tau, factor_ron_roff, mean_firing_rate, sampling_rate, duration)
current_neuron_PC = Barrel_PC('current', dt)
dynamic_neuron_PC = Barrel_PC('dynamic', dt)
current_neuron_IN = Barrel_IN('current', dt)
dynamic_neuron_IN = Barrel_IN('dynamic', dt)
current_neuron_PC.store()
dynamic_neuron_PC.store()
current_neuron_IN.store()
dynamic_neuron_IN.store()

for scale in scale_exc_inh:
    # PC
    current_neuron_PC.restore()
    dynamic_neuron_PC.restore()
    current_inj_PC = scale_input_theory(input_theory, 'current', baseline, scale, dt)
    print('CurrentPC')
    current_M_PC, current_S_PC = current_neuron_PC.run(current_inj_PC, duration, 35)
    current_inputs_PC[scale] = np.concatenate((current_inputs_PC[scale], current_M_PC.I_inj[0]/uA), axis=0)
    current_Vm_PC[scale] = np.concatenate((current_Vm_PC[scale], current_M_PC.v[0]/mV), axis=0)
    current_freq_PC[scale] = np.concatenate((current_freq_PC[scale], [current_S_PC.num_spikes/(duration/1000)]), axis=0) 
    dynamic_inj_PC = scale_input_theory(dynamic_theory, 'dynamic', baseline, scale, dt)
    print('DynamicPC')
    dynamic_M_PC, dynamic_S_PC = dynamic_neuron_PC.run(dynamic_inj_PC, duration, 35)
    dynamic_inputs_PC[scale] = np.concatenate((dynamic_inputs_PC[scale], dynamic_M_PC.I_inj[0]/uA), axis=0)
    dynamic_Vm_PC[scale] = np.concatenate((dynamic_Vm_PC[scale], dynamic_M_PC.v[0]/mV), axis=0)
    dynamic_freq_PC[scale] = np.concatenate((dynamic_freq_PC[scale], [dynamic_S_PC.num_spikes/(duration/1000)]), axis=0) 

    # IN
    current_neuron_IN.restore()
    dynamic_neuron_IN.restore()
    current_inj_IN = scale_input_theory(input_theory, 'current', baseline, scale, dt)
    print('CurrentIN')
    current_M_IN, current_S_IN = current_neuron_IN.run(current_inj_IN, duration, 11)
    current_inputs_IN[scale] = np.concatenate((current_inputs_IN[scale], current_M_IN.I_inj[0]/uA), axis=0)
    current_Vm_IN[scale] = np.concatenate((current_Vm_IN[scale], current_M_IN.v[0]/mV), axis=0)
    current_freq_IN[scale] = np.concatenate((current_freq_IN[scale], [current_S_IN.num_spikes/(duration/1000)]), axis=0) 
    dynamic_inj_IN = scale_input_theory(dynamic_theory, 'dynamic', baseline, scale, dt)
    print('DynamicIN')
    dynamic_M_IN, dynamic_S_IN = dynamic_neuron_IN.run(dynamic_inj_IN, duration, 11)
    dynamic_inputs_IN[scale] = np.concatenate((dynamic_inputs_IN[scale], dynamic_M_IN.I_inj[0]/uA), axis=0)
    dynamic_Vm_IN[scale] = np.concatenate((dynamic_Vm_IN[scale], dynamic_M_IN.v[0]/mV), axis=0)
    dynamic_freq_IN[scale] = np.concatenate((dynamic_freq_IN[scale], [dynamic_S_IN.num_spikes/(duration/1000)]), axis=0) 

    # Sanity Check
    # plot_dynamicclamp(dynamic_M, dynamic_input[0], dynamic_input[1], hidden_state, dt=dt)
    # plot_currentclamp(current_M, hidden_state, dt)
        
        
current_dict = {'PC':{'I':current_inputs_PC, 'Vm':current_Vm_PC, 'f':current_freq_PC}, 'IN':{'I':current_inputs_IN, 'Vm':current_Vm_IN, 'f':current_freq_IN}}
dynamic_dict = {'PC':{'I':dynamic_inputs_PC, 'Vm':dynamic_Vm_PC, 'f':dynamic_freq_PC}, 'IN':{'I':dynamic_inputs_IN, 'Vm':dynamic_Vm_IN, 'f':dynamic_freq_IN}}

# # Plot
# # plot_scaling_compare([current_dict, dynamic_dict])
# scale_array = dynamic_dict['I'].keys()
# N = len(scale_array)
# x = np.arange(N+1)

# fig, axs = plt.subplots(ncols=3, figsize=(15,8))
# sns.scatterplot(x=scale_array, y=[np.mean(i) for i in current_dict['I'].values()], color='red', ax=axs[0])
# sns.scatterplot(x=scale_array, y=[np.mean(i) for i in dynamic_dict['I'].values()], color='blue', ax=axs[0])
# sns.scatterplot(x=scale_array, y=[np.mean(i) for i in current_dict['Vm'].values()], color='red', ax=axs[1])
# sns.scatterplot(x=scale_array, y=[np.mean(i) for i in dynamic_dict['Vm'].values()], color='blue', ax=axs[1])
# sns.scatterplot(x=scale_array, y=[np.mean(i) for i in current_dict['f'].values()], color='red', ax=axs[2])
# sns.scatterplot(x=scale_array, y=[np.mean(i) for i in dynamic_dict['f'].values()], color='blue', ax=axs[2])

# # axs[0].set(xlabel='Input Current[uA]', ylabel='density')
# axs[0].title.set_text('Injected current')
# # axs[1].set(xlabel='Membrane potential [mV]')
# axs[1].title.set_text('Membrane potential')
# # axs[2].set(xticks=[], ylabel='Frequency [Hz]')
# axs[2].title.set_text('Firing frequency')
# fig.suptitle('Scaling effect')
# plt.legend(['Current', 'Dynamic'])
# plt.show()

# Save
np.save('results/saved/scaling_compare/current_dict.npy', current_dict)
np.save('results/saved/scaling_compare/dynamic_dict.npy', dynamic_dict)
