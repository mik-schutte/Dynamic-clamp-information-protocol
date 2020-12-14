'''
    make_input_experiments.py

    Make input current based on a artificial network responding to a hidden state
    The method is described in the following paper:
    Zeldenrust, F., de Knecht, S., Wadman, W. J., Denève, S., Gutkin, B., Knecht, S. De, Denève, S. (2017). 
    Estimating the Information Extracted by a Single Spiking Neuron from a Continuous Input Time Series. 
    Frontiers in Computational Neuroscience, 11(June), 49. doi:10.3389/FNCOM.2017.00049
    Please cite this reference when using this method.
    
    NB Make sure that you save the hidden state with the experiments, it is
    essential for the information calculation!
    
    Example use:
    [input_current, hidden_state] = make_input_experiments(0, 1000, 20/3000,40/3000, (0.5)/1000, 1, 20000);
    This creates an input current with baseline 0 pA, amplitude 1000 pA, tau=50 ms, the mean firing rate of neurons in the artificial network is 0.5 Hz, sampling rate of 1 kHz, 20000 ms long (you will need at least about 20 s for a good estimate. 
''' 
from code.input import Input
import numpy as np

def make_input_experiments(qon_qoff_type, baseline, amplitude_scaling, tau, factor_ron_roff, mean_firing_rate, sampling_rate, duration, seed=None):
    ''' Make input current based on a artificial network responding to a hidden state.

    INPUT:
        qon_qoff_type (str): The method of qon/qoff generation. Options are normal,
                             balanced and balanced_uniform.
        baseline (pA): Baseline for scaling the input current in picoampere.
        amplitude_scaling (pA): Scaling of the stimulus in picoampere
        tau (ms): Switching speed of the hidden state in milliseconds.
        mean_firing_rate (kHz): Mean firing rate of the artificial neurons in kilohertz.
        sampling rate (kHZ): Sampling rate of the experimental setup (injected current) 
                             in kilohertz.
        duration (ms): Length of the duration in milliseconds.
        seed (optional): Seed used in the random number generator.

    OUTPUT: 
        input_current: 1xN array with current values.
        input_theory: 1xN array with unscaled, theoretical input.
        hidden_state: 1xN array with hidden state values 0=off 1=on.
    '''
    # Set RNG seed, if no seed is provided
    if seed == None:
        np.random.seed()
        seed = np.random.randint(1000000000)

    # Fixed parameters
    N = 1000                            
    dt = 1./sampling_rate              #TODO Ask: Is alpha the same as alphan in create_qon_qoff()?
    tau_exponential_kernel = 5 
    alpha = np.sqrt(1/8) 
    stdq = alpha*mean_firing_rate
    ron = 1./(tau*(1+factor_ron_roff))
    roff = factor_ron_roff*ron

    #Create input from artifical network
    input_bayes = Input()
    input_bayes.dt = dt
    input_bayes.T = duration
    input_bayes.kernel = 'exponential'
    input_bayes.kerneltau = tau_exponential_kernel
    input_bayes.ron = ron
    input_bayes.roff = roff
    input_bayes.seed = seed
    input_bayes.xseed = seed        #TODO Ask: What's the difference between seed, xseed and qseed?

    # Create qon/qoff
    if qon_qoff_type == 'normal':
        mutheta = None                 #TODO Ask: What should mutheta look like?
        alphan = None
        regime = None
        [input_bayes.qon, input_bayes.qoff] = input_bayes.create_qonqoff(mutheta, N, alphan, regime, seed)
    elif qon_qoff_type == 'balanced':
        [input_bayes.qon, input_bayes.qoff] = input_bayes.create_qonqoff_balanced(N, mean_firing_rate, stdq, seed)
    elif qon_qoff_type == 'balanced_uniform':
        minq = 100                  #TODO Ask: What are good default minq/maxq values?
        maxq = 1000
        [input_bayes.qon, input_bayes.qoff] = input_bayes.create_qonqoff_balanced_uniform(N, minq, maxq, seed)
    else: 
        raise SyntaxError('No qon/qoff creation type specified')

    input_bayes.get_all()
    input_bayes.fHandle = input_bayes.markov()
    input_bayes.generate()

    #Scale for experiments
    input_current = amplitude_scaling*input_bayes.input+baseline
    input_theory = input_bayes.input
    hidden_state = input_bayes.x
    return [input_current, input_theory, hidden_state]