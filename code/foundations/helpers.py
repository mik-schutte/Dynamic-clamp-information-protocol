''' helpers.py
    
    Helps you out :)
''' 
import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np
from brian2 import *
from models.models import Barrel_PC, Barrel_IN

def scale_to_freq(neuron, input_theory, target, clamp_type, duration, dt=0.5, Ni=None):
    ''' docstring
    '''
    # Checks
    try:
        if neuron.stored == False:
            neuron.store()
    except:
        raise TypeError('Please insert a neuron class')
    if clamp_type != 'current' and clamp_type != 'dynamic':
        raise ValueError('ClampType must be \'current\' or \'dynamic\'')

    freq_diff = []
    scale_list =[1, 2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20, 22.5, 25]
    for idx, scale in enumerate(scale_list):
        neuron.restore()

        if clamp_type == 'current':
            inj = scale_input_theory(input_theory, 0, scale, dt)
        elif clamp_type == 'dynamic':
            inj = scale_dynamic_input(input_theory[0], input_theory[1], scale, dt)

        M, S = neuron.run(inj, duration*ms, Ni)
        diff = abs(S.num_spikes/(duration/1000) - target)
        freq_diff.append(diff)

        # Check if prior scale wasn't a better fit
        if freq_diff[idx-1] < diff:
            print('scale to use=', scale_list[idx-1])
            neuron.restore()
            return scale_list[idx-1]

    print('Best scale was:', scale_list[-1])
    neuron.restore()
    return scale_list[-1]
    
def scale_input_theory(input_theory, baseline, amplitude_scaling, dt):
    ''' docstring
    '''
    scaled_input = (baseline + input_theory * amplitude_scaling)*uamp
    inject_input = TimedArray(scaled_input, dt=dt*ms)
    return inject_input

def scale_dynamic_input(g_exc, g_inh, scale_exc_inh, dt):
    ''' docstring
    '''
    g_exc = g_exc*mS * scale_exc_inh
    g_inh = g_inh*mS * scale_exc_inh
    g_exc = TimedArray(g_exc, dt=dt*ms)
    g_inh = TimedArray(g_inh, dt=dt*ms)
    return (g_exc, g_inh)

def make_spiketrain(S, hiddenstate, dt):
    ''' docstring
    '''
    spiketrain = np.zeros((1, hiddenstate.shape[0]))
    spikeidx = np.array(S.t/ms/dt, dtype=int)
    spiketrain[:, spikeidx] = 1
    return spiketrain

def get_spike_intervals(S):
    # Check
    if not isinstance(S, StateMonitor):
        TypeError('No SpikeMonitor provided')
        
    intervals = []
    for i in range(len(S.t)-1):
        intervals.append(abs(S.t[i+1] - S.t[i])/ms)
    return intervals
