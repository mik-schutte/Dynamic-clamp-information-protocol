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

def scale_to_freq(neuron, input_theory, target, on_all_ratio, clamp_type, duration, hidden_state, scale_list, dt=0.5, Ni=None):
    ''' Scales the theoretical input to an input that results in target firing frequence 
        by running test simulations. 

        INPUT
        neuron (Class): neuron model as found in models.py
        input_theory (array or tuple): theoretical input that has to be scaled; (g_exc, g_inh) if dynamic
        target (int): target frequency for the simulation
        on_all_ratio (float): how much more does the neuron need to fire during the ON state
        clamp_type (str): 'current' or 'dynamic' 
        duration (int): duration of the simulation in miliseconds
        hidden_state (array): binary array representing the hidden state
        dt (float): time step of the simulation and hiddenstate
        Ni (int): index of the neuron to be simulated

        OUTPUT
        inj_input (brian2.TimedArray): the input that results in the target firing frequency
    '''
    # Checks
    try:
        if neuron.stored == False:
            neuron.store()
    except:
        raise TypeError('Please insert a neuron class')
    if clamp_type != 'current' and clamp_type != 'dynamic':
        raise ValueError('ClampType must be \'current\' or \'dynamic\'')
    if clamp_type == 'current':
        if len(input_theory) != len(hidden_state):
            raise  AssertionError('Input and hidden state don\'t correspond')
    elif clamp_type == 'dynamic':
        if len(input_theory[0]) != len(hidden_state):
            raise  AssertionError('Input and hidden state don\'t correspond')
    
    freq_diff_list = []  # Containing the difference between target and actual frequency
    freq_list = []       # Containing the actual frequencies
    on_freq_list = []    # Containing the frequency during ON-state

    for idx, scale in enumerate(scale_list):
        neuron.restore()

        # Scale and run
        inj = scale_input_theory(input_theory, clamp_type, 0, scale, dt)
        M, S = neuron.run(inj, duration, Ni)

        # Compare against frequency target
        freq = S.num_spikes/(duration/1000)
        freq_list.append(freq)
        freq_diff = abs(freq - target)
        freq_diff_list.append(freq_diff)

        # Compare against on_frequency target
        spiketrain = make_spiketrain(S, duration, dt)
        on_freq = get_on_freq(spiketrain, hidden_state, dt)
        on_freq_list.append(on_freq)

        if freq > target and idx != 0:
            # Check if prior or current scale is a better fit
            if freq_diff_list[idx-1] <= freq_diff_list[idx]:
                ideal = idx - 1
            else:
                ideal = idx

            # Check ON/OFF ratio
            if on_freq_list[ideal]/freq_list[ideal] >= on_all_ratio:
                neuron.restore()
                print(scale_list[ideal], freq_list[ideal])
                return scale_input_theory(input_theory, clamp_type, 0, scale_list[ideal], dt)
            else:
                neuron.restore()
                return False
     
    # When all scales have been tried
    # Check for ON/All ratio
    if on_freq/freq < on_all_ratio:
        neuron.restore()
        return False

    neuron.restore()
    print(scale_list[-1], freq_list[-1])
    return scale_input_theory(input_theory, clamp_type, 0, scale_list[-1], dt)
    
def scale_input_theory(input_theory, clamp_type, baseline, scale, dt):
    ''' Scales the theoretical current or dynamic input with a scale factor. 
        
        INPUT
        input_theory (array or tuple): array of theoretical current or (g_exc, g_inh)
        clamp_type (str): 'current' or 'dynamic'
        baseline (float or tuple): baseline if dynamic (base_exc, base_inh)
        scale (float): scaling factor
        dt (float): time step of the simulation

        OUTPUT
        inj_input (brian2.TimedArray): the scaled input
    '''
    if clamp_type == 'current':
        baseline = np.ones_like(input_theory, dtype=float)*baseline
        scaled_input = (baseline + input_theory * scale)*uamp
        inject_input = TimedArray(scaled_input, dt=dt*ms)

    elif clamp_type == 'dynamic':
        g_exc, g_inh = input_theory
        g_exc = (baseline + g_exc * scale)*mS
        g_inh = (baseline + g_inh * scale)*mS
        g_exc = TimedArray(g_exc, dt=dt*ms)
        g_inh = TimedArray(g_inh, dt=dt*ms)
        inject_input = (g_exc, g_inh)
       
    return inject_input

def make_spiketrain(spikemon, duration, dt):
    ''' Generates a binary array that spans the whole simulation and 
        is 1 when a spike is fired.
    '''
    spiketrain = np.zeros((1, int(duration/dt)))
    if isinstance(spikemon, SpikeMonitor):
        spikeidx = np.array(spikemon.t/ms/dt, dtype=int)
    elif isinstance(spikemon, (np.ndarray, list)):
        spikeidx = spikemon/dt
        spikeidx = spikeidx.astype('int')
    else:
        TypeError('Please provide SpikeMonitor or array of spiketimes')
    spiketrain[:, spikeidx] = 1
    return spiketrain

def get_spike_intervals(spikemon):
    ''' Determine the interval between spikes in milliseconds. 
    '''
    intervals = []
    # Check
    if isinstance(spikemon, SpikeMonitor):
        for i in range(len(spikemon.t)-1):
            intervals.append(abs(spikemon.t[i+1] - spikemon.t[i])/ms)
    elif isinstance(spikemon, (np.ndarray, list)):
        for i in range(len(spikemon)-1):
            intervals.append(abs(spikemon[i+1] - spikemon[i]))
    else:
        TypeError('Please provide SpikeMonitor or array of spiketimes')
    return intervals


def get_on_index(hidden_state):
    ''' Get the indexis where the hidden state is ON.
    '''
    # Index where hidden state is ON
    on_idx = []
    for idx, val in enumerate(hidden_state):
        if val == 1:
            on_idx.append(idx)
    return on_idx

def get_off_index(hidden_state):
    ''' Get the indexis where the hidden state is OFF.
    '''
    # Index where hidden state is OFF
    off_idx = []
    for idx, val in enumerate(hidden_state):
        if val == 0:
            off_idx.append(idx)
    return off_idx

def get_on_spikes(spiketrain, hidden_state):
    ''' Get the index where spikes are fired during ON-state.
    '''
    # ON index
    on_idx = get_on_index(hidden_state)

    # Spike index
    spike_idx = np.where(spiketrain==1)[1]

    # Spikes when hidden state is ON
    on_spikes = []
    for idx in spike_idx:
        if idx in on_idx:
            on_spikes.append(idx)
    return np.array(on_spikes)

def get_off_spikes(spiketrain, hidden_state):
    ''' Get the index where the spikes are fired during OFF-state.
    '''
    # OFF index
    off_idx = get_off_index(hidden_state)

    # Spike index
    spike_idx = np.where(spiketrain==1)[1]

    # Spike when hidden state is OFF
    off_spikes = []
    for idx in spike_idx:
        if idx in off_idx:
            off_spikes.append(idx)
    return np.array(off_spikes)

def get_on_freq(spiketrain, hidden_state, dt):
    ''' Get firing frequency during on state in Hertz (Hz).
    '''
    on_spike_count = len(get_on_spikes(spiketrain, hidden_state))
    on_duration = len(get_on_index(hidden_state))*dt*ms
    return (on_spike_count/on_duration)/Hz

def get_on_off_isi(spikemon, hidden_state, dt):
    ''' Get the ISI of the ON and OFF state.

        OUPUT
        ON_isi, OFF_isi
    '''
    if isinstance(spikemon, SpikeMonitor):
        spikemon = spikemon.t/ms
    spikemon = np.round(spikemon, 2)
    
    # Get hidden state indexes where a switch of state occurred
    diff = np.diff(hidden_state)
    switches = np.array([-1])
    for idx, value in enumerate(diff):
        if value != 0:
            switches = np.append(switches, idx)
    switches = np.append(switches, len(hidden_state) - 1)

    # Seperate the hidden state in to blocks
    block_array = []
    for idx, switch in enumerate(switches[:-1]):        
        start = switches[idx] + 1
        stop = switches[idx + 1]
        state = hidden_state[start]
        block = (start, stop, state)
        block_array.append(block)

    # Check if a spike occured during a block
    spikes_per_block = {1:[], 0:[]}
    for block in block_array:
        block_spikes = []
        block_time = np.arange(block[0]*dt, block[1]*dt+dt, dt)
        for t in block_time:
            t = np.round(t, 2)
            if t in spikemon:
                block_spikes.append(t)
        spikes_per_block[block[2]].append(block_spikes)

    # Get the ISI for spikes in each block
    on_isi = np.array([])
    off_isi = np.array([])
    for state in spikes_per_block:
        for block in spikes_per_block[state]:
            if state == 1:
                on_isi = np.append(on_isi, np.diff(block))
            elif state == 0:
                off_isi = np.append(off_isi, np.diff(block))
    
    return on_isi, off_isi