import numpy as np
from brian2 import *
from code.make_dynamic_experiments import make_dynamic_experiments
from models.models import *
from visualization.plotter import plot_currentclamp, plot_dynamicclamp, plot_compare
from code.MI_calculation import analyze_exp
'''Docstring
'''
def make_spiketrain(S, hiddenstate, dt):
    'Makes spiketrains'
    spiketrain = np.zeros((1, hiddenstate.shape[0]))
    spikeidx = np.array(S.t/ms/dt, dtype=int)
    spiketrain[:, spikeidx] = 1
    return spiketrain

## Generate input and hiddenstate
# Set parameters
baseline = 0  
amplitude_scaling = 25
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
N_runs = 61 # for all pyramidal parameters

# Generate input
print('Generating...')
[exc_LUT, inh_LUT, input_theory, hidden_state] = make_dynamic_experiments(qon_qoff_type, baseline, amplitude_scaling, tau, factor_ron_roff, mean_firing_rate, sampling_rate, duration, dv)
print('Input and hiddenstate generate!')

## Create and scale input to a Brian2 TimedArray
# Currentclamp
input_currentx = baseline + input_theory*amplitude_scaling
input_current = input_currentx*uamp
input_current = TimedArray(input_current, dt=dt*ms)

# Dynamicclamp
g_exc = abs(exc_LUT[-65] / (-65 - Er_exc))*mS * dynamic_scaling
g_inh = abs(inh_LUT[-65] / (-65 - Er_inh))*mS 
g_exc = TimedArray(g_exc, dt=dt*ms)
g_inh = TimedArray(g_inh, dt=dt*ms)

## Simulate
MI = {'current' : [], 'dynamic' : []}
print('Running simulation...')
for i in range(N_runs):
    print('Run', i ,'of', N_runs)
    # Clamps
    M_current, S_current = simulate_barrel_PC(input_current, duration*ms, 'current', Ni=i)
    M_dynamic, S_dynamic = simulate_barrel_PC((g_exc, g_inh), duration*ms, 'dynamic', Ni=i)

    # Create spiketrain
    spiketrain_current = make_spiketrain(S_current, hidden_state, dt)
    spiketrain_dynamic = make_spiketrain(S_dynamic, hidden_state, dt)

    # Calculate MI
    Output_current = analyze_exp(ron, roff, hidden_state, input_theory, dt, theta, spiketrain_current)
    Output_dynamic = analyze_exp(ron, roff, hidden_state, input_theory, dt, theta, spiketrain_dynamic)
    MI['current'].append(Output_current)
    MI['dynamic'].append(Output_dynamic)

    # Sanity check
    print(Output_dynamic['MI'])
    plot_dynamicclamp(M_dynamic, g_exc, g_inh, hidden_state, dt=dt)
    print(Output_current['MI'])
    plot_currentclamp(M_current, hidden_state, dt=dt)
    


print('Simulation complete, saving files')

# Save files
# np.savetxt(f'results/hiddenstate.csv', hidden_state, delimiter=',')
# np.savetxt(f'results/input_theory.csv', input_theory, delimiter=',')
# np.savetxt(f'results/spiketrain_current.csv', spiketrain_current, delimiter=',')
# np.savetxt(f'results/spiketrain_dynamic.csv', spiketrain_dynamic, delimiter=',')

# np.save(f'results/MI.npy', MI)


