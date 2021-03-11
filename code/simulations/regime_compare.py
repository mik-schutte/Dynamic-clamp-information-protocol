'''
    regime_compare.py
    
    Compare different regimes, as show in table 1 of Zeldenrust et al., (2017)
'''
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np
import pandas as pd
import scipy.stats as stats
from brian2 import *
from foundations.make_dynamic_experiments import make_dynamic_experiments
from foundations.MI_calculation import analyze_exp
from foundations.helpers import scale_input_theory
from models.models import Barrel_PC
from visualization.plotter import plot_currentclamp, plot_regime_compare

# Define different regimes
## Dict {regime : (tau, uq)}
regimes = {'slow' : (50, (0.5)/1000), 'fast' : (10, (2.5)/1000), 
            'slow_high' : (50, (2.5)/1000), 'fast_low' : (10, (0.5)/1000)}

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
dt = 1/sampling_rate 
dv = 0.5
duration = 2000
qon_qoff_type = 'balanced'
Er_exc, Er_inh = (0, -90)
N_runs = 10 

slow_membrane_potential, slow_inp = ([], [])
fast_membrane_potential, fast_inp = ([], [])
slow_high_membrane_potential, slow_high_inp = ([], [])
fast_low_membrane_potential, fast_low_inp = ([], [])
indexlist = ['g_exc', 'g_inh', 'input_theory', 'hidden_state']

# Initialize neurons for simulation
slow_neuron = Barrel_PC('current', dt=dt)
fast_neuron = Barrel_PC('current', dt=dt)
slow_high_neuron = Barrel_PC('current', dt=dt)
fast_low_neuron = Barrel_PC('current', dt=dt)
slow_neuron.store()
fast_neuron.store()
slow_high_neuron.store()
fast_low_neuron.store()
    
print('Comparing regimes')
for i in range(N_runs): 
    print('Round', i+1, 'of', N_runs)
    # Generate input
    slow_input = pd.Series(data=make_dynamic_experiments(qon_qoff_type, baseline, 
                    amplitude_scaling, regimes['slow'][0], factor_ron_roff, regimes['slow'][1],
                    sampling_rate, duration, dv), index=indexlist)
    fast_input = pd.Series(data=make_dynamic_experiments(qon_qoff_type, baseline, 
                    amplitude_scaling, regimes['fast'][0], factor_ron_roff, regimes['fast'][1],
                    sampling_rate, duration, dv), index=indexlist)
    slow_high_input = pd.Series(data=make_dynamic_experiments(qon_qoff_type, baseline, 
                    amplitude_scaling, regimes['slow_high'][0], factor_ron_roff, regimes['slow_high'][1],
                    sampling_rate, duration, dv), index=indexlist)
    fast_low_input = pd.Series(data=make_dynamic_experiments(qon_qoff_type, baseline, 
                    amplitude_scaling, regimes['fast_low'][0], factor_ron_roff, regimes['fast_low'][1],
                    sampling_rate, duration, dv), index=indexlist)

    ## Create and scale input to a Brian2 TimedArray
    slow_input_current = scale_input_theory(slow_input['input_theory'], baseline, amplitude_scaling, dt)
    fast_input_current = scale_input_theory(fast_input['input_theory'], baseline, amplitude_scaling, dt)
    slow_high_input_current = scale_input_theory(slow_high_input['input_theory'], baseline, amplitude_scaling, dt)
    fast_low_input_current = scale_input_theory(fast_low_input['input_theory'], baseline, amplitude_scaling, dt)
    
    # Simulate
    slow_neuron.restore()
    fast_neuron.restore()
    slow_high_neuron.restore()
    fast_low_neuron.restore()
    
    slow_M, slow_S = slow_neuron.run(slow_input_current, duration*ms, Ni=1, Er_exc, Er_inh)
    fast_M, fast_S = fast_neuron.run(fast_input_current, duration*ms, Ni=1, Er_exc, Er_inh)
    slow_high_M, slow_high_S = slow_high_neuron.run(slow_high_input_current, duration*ms, Ni=1, Er_exc, Er_inh)
    fast_low_M, fast_low_S = fast_low_neuron.run(fast_low_input_current, duration*ms, Ni=1, Er_exc, Er_inh)

    slow_membrane_potential = np.concatenate((slow_membrane_potential, slow_M.v[0]/mV), axis=0) 
    fast_membrane_potential= np.concatenate((fast_membrane_potential, fast_M.v[0]/mV), axis=0)
    slow_high_membrane_potential = np.concatenate((slow_high_membrane_potential, slow_high_M.v[0]/mV), axis=0) 
    fast_low_membrane_potential= np.concatenate((fast_low_membrane_potential, fast_low_M.v[0]/mV), axis=0)

    slow_inp = np.concatenate((slow_inp, slow_M.I_inj[0]/nA), axis=0) 
    fast_inp= np.concatenate((fast_inp, fast_M.I_inj[0]/nA), axis=0)
    slow_high_inp = np.concatenate((slow_high_inp, slow_high_M.I_inj[0]/nA), axis=0) 
    fast_low_inp= np.concatenate((fast_low_inp, fast_low_M.I_inj[0]/nA), axis=0)

    # # Sanity Check
    # plot_currentclamp(slow_M, slow_input['hidden_state'], dt)
    # plot_currentclamp(slow_high_M, slow_high_input['hidden_state'], dt)
    # plot_currentclamp(fast_M, fast_input['hidden_state'], dt)
    # plot_currentclamp(fast_low_M, fast_low_input['hidden_state'], dt)

slow = {'potential' : slow_membrane_potential, 'input' : slow_inp}
fast = {'potential' : fast_membrane_potential, 'input' : fast_inp}
slow_high = {'potential' : slow_high_membrane_potential, 'input' : slow_high_inp}
fast_low = {'potential' : fast_low_membrane_potential, 'input' : fast_low_inp}

# Plot
plot_regime_compare([slow, fast, slow_high, fast_low])

# Save
# np.save(f'results/saved/regime_compare/slow.npy', slow) 
# np.save(f'results/saved/regime_compare/slow_high.npy', slow_high)  
# np.save(f'results/saved/regime_compare/fast.npy', fast)
# np.save(f'results/saved/regime_compare/fast_low.npy', fast_low)