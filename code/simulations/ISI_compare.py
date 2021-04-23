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
from visualization.plotter import plot_currentclamp, plot_dynamicclamp, plot_ISI_compare
import seaborn as sns
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
on_off_ratio = 1.5
N_runs = (61, 22) # for all pyramidal and interneuron parameters

# Create ISI dict
keys = ['current_PC', 'dynamic_PC', 'current_IN', 'dynamic_IN']
ISI = dict.fromkeys(keys)
for key in ISI.keys():
    ISI[key] = {'on':[], 'off':[]}

current_PC = Barrel_PC('current', dt)
dynamic_PC = Barrel_PC('dynamic', dt)
current_IN = Barrel_IN('current', dt)
dynamic_IN = Barrel_IN('dynamic', dt)
current_PC.store()
dynamic_PC.store()
current_IN.store()
dynamic_IN.store()

print('Running simulation') 
for _ in range(5):
    # Generate input
    [g_exc, g_inh, input_theory, hidden_state] = make_dynamic_experiments(qon_qoff_type, baseline, amplitude_scaling, tau, factor_ron_roff, mean_firing_rate, sampling_rate, duration, dv)
    
    # Pyramidal Cells
    i = 35
    current_PC.restore()
    dynamic_PC.restore()
    
    # Scale input
    inj_current = scale_to_freq(current_PC, input_theory, target, on_off_ratio, 'current', duration, hidden_state, dt, i)
    inj_dynamic = scale_to_freq(dynamic_PC, (g_exc, g_inh), target, on_off_ratio, 'dynamic', duration, hidden_state, dt, i)

    # Simulate and calculate ISI
    if inj_current != False and inj_dynamic != False:
        current_M, current_S = current_PC.run(inj_current, duration*ms, Ni=i)
        dynamic_M, dynamic_S = dynamic_PC.run(inj_dynamic, duration*ms, Ni=i)
        ISI_current_on, ISI_current_off = get_on_off_isi(hidden_state, current_S, dt)
        ISI_dynamic_on, ISI_dynamic_off = get_on_off_isi(hidden_state, dynamic_S, dt)
        ISI['current_PC']['on'] = np.append(ISI['current_PC']['on'], ISI_current_on)
        ISI['current_PC']['off'] = np.append(ISI['current_PC']['off'], ISI_current_off)
        ISI['dynamic_PC']['on'] = np.append(ISI['dynamic_PC']['on'], ISI_dynamic_on)
        ISI['dynamic_PC']['off'] = np.append(ISI['dynamic_PC']['off'], ISI_dynamic_off)

        # # Sanity Check:
        # plot_currentclamp(current_M, hidden_state, dt)
        # plot_dynamicclamp(dynamic_M, inj_dynamic[0], inj_dynamic[1], hidden_state, dt)

    # Interneurons, 
    i = 11
    current_IN.restore()
    dynamic_IN.restore()

    # Scale input
    inj_current = scale_to_freq(current_IN, input_theory, target, on_off_ratio, 'current', duration, hidden_state, dt, i)
    inj_dynamic = scale_to_freq(dynamic_IN, (g_exc, g_inh), target, on_off_ratio, 'dynamic', duration, hidden_state, dt, i)

    # inj_current = scale_input_theory(input_theory, baseline, current_scale, dt)
    # inj_dynamic = scale_dynamic_input(g_exc, g_inh, dynamic_scale, dt)

    # Simulate and calculate
    if inj_current != False and inj_dynamic != False:
        current_M, current_S = current_IN.run(inj_current, duration*ms, Ni=i)
        dynamic_M, dynamic_S = dynamic_IN.run(inj_dynamic, duration*ms, Ni=i)
        ISI_current_on, ISI_current_off = get_on_off_isi(hidden_state, current_S, dt)
        ISI_dynamic_on, ISI_dynamic_off = get_on_off_isi(hidden_state, dynamic_S, dt)
        ISI['current_IN']['on'] = np.append(ISI['current_IN']['on'], ISI_current_on)
        ISI['current_IN']['off'] = np.append(ISI['current_IN']['off'], ISI_current_off)
        ISI['dynamic_IN']['on'] = np.append(ISI['dynamic_IN']['on'], ISI_dynamic_on)
        ISI['dynamic_IN']['off'] = np.append(ISI['dynamic_IN']['off'], ISI_dynamic_off)

        # # Sanity Check:
        # plot_currentclamp(current_M, hidden_state, dt)
        # plot_dynamicclamp(dynamic_M, inj_dynamic[0], inj_dynamic[1], hidden_state, dt)

# Save ISI dictionary
np.save(f'results/saved/ISI_compare/ISI_test.npy', ISI)

# Clear Brian2 cache
try:
    clear_cache('cython')
except:
    pass

# # Plot
# fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
# sns.histplot(ISI['current_PC']['on'], ax=axs[0, 0], kde=True, kde_kws={'bw_adjust':2.5}, bins=50, color='red')
# sns.histplot(ISI['current_PC']['off'], ax=axs[0, 1], kde=True, kde_kws={'bw_adjust':2.5}, bins=50, color='salmon')
# sns.histplot(ISI['dynamic_PC']['on'], ax=axs[1, 0], kde=True, kde_kws={'bw_adjust':2.5}, bins=50, color='blue')
# sns.histplot(ISI['dynamic_PC']['off'], ax=axs[1, 1], kde=True, kde_kws={'bw_adjust':2.5}, bins=50, color='steelblue')

# # axs[0, 0].set_yscale('log')
# # axs[0, 1].set_yscale('log')
# # axs[1, 0].set_yscale('log')
# # axs[1, 1].set_yscale('log')

# axs[0, 0].title.set_text('current ON')
# axs[0, 1].title.set_text('current OFF')
# axs[1, 0].title.set_text('dynamic ON')
# axs[1, 1].title.set_text('dynamic OFF')
# fig.suptitle('Pyramidal Cell ISI')
# plt.show()