'''
    make_input_experiments.py
    ADDITIONAL INFORMATION HERE
''' 
from code.input import Input
import numpy as np

def make_input_experiments(baseline, amplitude_scaling, tau, factor_ron_roff, mean_firing_rate, sampling_rate, duration, seed=None):
    '''Doc
    '''

    # Parameters
    if seed == None:
        np.random.seed()
        seed = np.random.rand()
    
    dt = 1./sampling_rate

    # Fixed parameters
    N = 1000
    tau_exponential_kernel = 5 #ms
    alpha = np.sqrt(1/8) # relation mean and std of the firing rates of the artificial neurons (firing rates should be positive)
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
    input_bayes.xseed = seed
    [input_bayes.qon, input_bayes.qoff] = input_bayes.create_qonqoff_balanced(N, mean_firing_rate, stdq, seed)
    input_bayes.get_all()
    input_bayes.fHandle = input_bayes.markov()
    input_bayes.generate()

    #Scale for experiments
    input_current = amplitude_scaling*input_bayes.input+baseline
    input_theory = input_bayes.input
    hidden_state = input_bayes.x

    return [input_current, input_theory, hidden_state]