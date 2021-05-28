''' big_sim.py
'''

import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from foundations.helpers import scale_input_theory
from brian2 import clear_cache, uA, mV, ms
from models.models import Barrel_PC, Barrel_IN
from foundations.helpers import scale_to_freq
from foundations.make_dynamic_experiments import make_dynamic_experiments
import numpy as np
import pandas as pd

# Set Parameters
baseline = 0  
theta = 0     
factor_ron_roff = 2    
tau_PC = 250
ron_PC = 1./(tau_PC*(1+factor_ron_roff))
roff_PC = factor_ron_roff*ron_PC
mean_firing_rate_PC = (0.1)/1000  
duration_PC = 100000
tau_IN = 50               
ron_IN = 1./(tau_IN*(1+factor_ron_roff))
roff_IN = factor_ron_roff*ron_IN
mean_firing_rate_IN = (0.5)/1000
duration_IN = 20000 
sampling_rate = 5      
dt = 1/sampling_rate 
qon_qoff_type = 'balanced'
Er_exc, Er_inh = (0, -75)
scales = {'CC_PC':19, 'DC_PC':30, 'CC_IN':17, 'DC_IN':6}
N_runs = 20


# Initiate Pyramidal cell models
PC_i = 35
current_PC = Barrel_PC('current', dt=dt)
dynamic_PC = Barrel_PC('dynamic', dt=dt)
current_PC.store()
dynamic_PC.store()

# Create results DataFrame
vars_to_track = ['input_theory', 'dynamic_theory', 'hidden_state',
                 'inj_current', 'current_volt', 'current_spikes',
                 'inj_dynamic', 'dynamic_volt', 'dynamic_spikes', 'dynamic_g']
results_PC = pd.DataFrame(columns=vars_to_track)

# Pyramidal Cell simulation
for _ in range(N_runs):
    # Make input theory and hidden state for Pyramidal Cell
    [input_theory, dynamic_theory, hidden_state] = make_dynamic_experiments(qon_qoff_type, baseline, tau_PC, factor_ron_roff, mean_firing_rate_PC, sampling_rate, duration_PC)

    # Scale input
    inj_current = scale_input_theory(input_theory, 'current', 0, scales['CC_PC'], dt)
    inj_dynamic = scale_input_theory(dynamic_theory, 'dynamic', 0, scales['DC_PC'], dt)
  
    # Run Pyramidal Cell
    current_PC.restore()
    dynamic_PC.restore()
    current_PC_M, current_PC_S = current_PC.run(inj_current, duration_PC, PC_i)
    dynamic_PC_M, dynamic_PC_S = dynamic_PC.run(inj_dynamic, duration_PC, PC_i)

    # Store results       
    data = np.array([input_theory, dynamic_theory, hidden_state,
                    current_PC_M.I_inj[0]/uA, current_PC_M.v[0]/mV, current_PC_S.t/ms,
                    dynamic_PC_M.I_inj[0]/uA, dynamic_PC_M.v[0]/mV, dynamic_PC_S.t/ms, (inj_dynamic[0].values, inj_dynamic[1].values)], dtype=list)
    data = pd.DataFrame(data=data, index=vars_to_track).T
    results_PC = results_PC.append(data, ignore_index=True)

# Keep it clean
try:
    clear_cache('cython')
except:
    pass

# Initiate Interneuron models
IN_i = 11
current_IN = Barrel_IN('current', dt=dt)
dynamic_IN = Barrel_IN('dynamic', dt=dt)
current_IN.store()
dynamic_IN.store()

# Create results DataFrame
results_IN = pd.DataFrame(columns=vars_to_track)

# Interneuron simulation
for _ in range(N_runs):
    # Make iput theory and hidden state for interneurons
    [input_theory, dynamic_theory, hidden_state] = make_dynamic_experiments(qon_qoff_type, baseline, tau_IN, factor_ron_roff, mean_firing_rate_IN, sampling_rate, duration_IN)

    # Scale input
    inj_current = scale_input_theory(input_theory, 'current', 0, scales['CC_IN'], dt)
    inj_dynamic= scale_input_theory(dynamic_theory, 'dynamic', 0, scales['DC_IN'], dt)

    # Run Interneurons 
    current_IN.restore()
    dynamic_IN.restore()
    current_IN_M, current_IN_S = current_IN.run(inj_current, duration_IN, IN_i)
    dynamic_IN_M, dynamic_IN_S = dynamic_IN.run(inj_dynamic, duration_IN, IN_i)

    # Store results       
    data = np.array([input_theory, dynamic_theory, hidden_state,
                    current_IN_M.I_inj[0]/uA, current_IN_M.v[0]/mV, current_IN_S.t/ms,
                    dynamic_IN_M.I_inj[0]/uA, dynamic_IN_M.v[0]/mV, dynamic_IN_S.t/ms, (inj_dynamic[0].values, inj_dynamic[1].values)], dtype=list)
    data = pd.DataFrame(data=data, index=vars_to_track).T
    results_IN = results_IN.append(data, ignore_index=True)

# Save data
results_PC.to_pickle('results/results_PC.pkl')
results_IN.to_pickle('results/results_IN.pkl')

# Clean cache
try:
    clear_cache('cython')
except:
    pass
