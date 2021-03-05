''' dt_compare.py
    
    Compares input and simulation results with different integration steps (dt)
'''
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from foundations.make_dynamic_experiments import make_dynamic_experiments
from foundations.helpers import scale_input_theory
from models.models import Barrel_PC, Barrel_IN
from visualization.plotter import plot_dt_compare
from brian2 import *
from visualization.plotter import plot_currentclamp
import numpy as np
import scipy.stats as stats

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
dv = 0.5
duration = 2000
qon_qoff_type = 'balanced'
Er_exc, Er_inh = (0, -75)
N_runs = 10 # for all pyramidal and interneuron parameters

sampling_array = [20, 10, 5, 2]
PC_results_I = dict.fromkeys(sampling_array, [])
PC_results_Vm = dict.fromkeys(sampling_array, [])
IN_results_I = dict.fromkeys(sampling_array, [])
IN_results_Vm = dict.fromkeys(sampling_array, [])

print('Running simulation...')
for _ in range(N_runs):
    for sampling_rate in sampling_array:
        # Generate input
        [g_exc, g_inh, input_theory, hidden_state] = make_dynamic_experiments(qon_qoff_type, baseline, amplitude_scaling, tau, factor_ron_roff, mean_firing_rate, sampling_rate, duration, dv)

        # Scale input
        inj_input = scale_input_theory(input_theory, baseline, amplitude_scaling, 1/sampling_rate)

        # Initialize neuron
        PC = Barrel_PC(clamp_type='current', dt=1/sampling_rate)
        IN = Barrel_IN(clamp_type='current', dt=1/sampling_rate)

        # Run simulation
        PC_M, PC_S = PC.run(inj_input, duration*ms, Ni=1)
        IN_M, IN_S = IN.run(inj_input, duration*ms, Ni=1)
        
        # Store results
        PC_results_I[sampling_rate] = np.concatenate((PC_results_I[sampling_rate], PC_M.I_inj[0]/nA), axis=0) 
        PC_results_Vm[sampling_rate] = np.concatenate((PC_results_Vm[sampling_rate], PC_M.v[0]/mV), axis=0) 
        IN_results_I[sampling_rate] = np.concatenate((IN_results_I[sampling_rate], IN_M.I_inj[0]/nA), axis=0) 
        IN_results_Vm[sampling_rate] = np.concatenate((IN_results_Vm[sampling_rate], IN_M.v[0]/mV), axis=0) 

PC_results = {'I' : PC_results_I, 'Vm' : PC_results_Vm}
IN_results = {'I' : IN_results_I, 'Vm' : IN_results_Vm}
plot_dt_compare([PC_results, IN_results])

# Save
# np.save(f'results/saved/dt_compare/PC_results.npy', {'I' : PC_results_I, 'Vm' : PC_results_Vm})
# np.save(f'results/saved/dt_compare/IN_results.npy', {'I' : IN_results_I, 'Vm' : IN_results_Vm})