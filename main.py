from code.input import Input
from code.make_input_experiments import make_input_experiments
import numpy as np


baseline = 0           
amplitude = 700      
tau = 50               
factor_ron_roff = 2    
mean_firing_rate = (0.5)/1000 
sampling_rate = 5      
dt = 1/sampling_rate 
duration = 20000
qon_qoff_type = 'balanced'

print('Running...')
[input_current, input_theory, hidden_state] = make_input_experiments(qon_qoff_type, baseline, amplitude, tau, factor_ron_roff, mean_firing_rate, sampling_rate, duration)
print('Done')

# Save
np.savetxt(f'results/hiddenstate.csv', hidden_state, delimiter=',')
np.savetxt(f'results/input_current.csv', input_current, delimiter=',')
np.savetxt(f'results/input_theory.csv', input_theory, delimiter=',')