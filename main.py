'''
    main.py

    This file is the main python file.
    The method is described in the following paper:
    Zeldenrust, F., de Knecht, S., Wadman, W. J., Denève, S., Gutkin, B., Knecht, S. De, Denève, S. (2017). 
    Estimating the Information Extracted by a Single Spiking Neuron from a Continuous Input Time Series. 
    Frontiers in Computational Neuroscience, 11(June), 49. doi:10.3389/FNCOM.2017.00049
    Please cite this reference when using this method.
'''
from code.input import Input
from code.make_input_experiments import make_input_experiments
import numpy as np

# Set parameters
baseline = 0           
amplitude_scaling = 700      
tau = 50               
factor_ron_roff = 2    
mean_firing_rate = (0.5)/1000 
sampling_rate = 5      
dt = 1/sampling_rate 
duration = 20000
qon_qoff_type = 'normal'

# Run
print('Running...')
[input_current, input_theory, hidden_state] = make_input_experiments(qon_qoff_type, baseline, amplitude_scaling, tau, factor_ron_roff, mean_firing_rate, sampling_rate, duration)
print('Done')

# Save 
np.savetxt(f'results/hiddenstate.csv', hidden_state, delimiter=',')
np.savetxt(f'results/input_current.csv', input_current, delimiter=',')
np.savetxt(f'results/input_theory.csv', input_theory, delimiter=',')