'''
    make_dynamic_experiments.py

'''
import numpy as np
from code.input import Input
from code.dynamic_clamp import get_g0, get_input_LUT 

def make_dynamic_experiments(qon_qoff_type, baseline, amplitude_scaling, tau, factor_ron_roff, mean_firing_rate, sampling_rate, duration, dv, seed=None):
    '''docstring
    '''
    
    # Set RNG seed, if no seed is provided
    if seed == None:
        np.random.seed()
        seed = np.random.randint(1000000000)

    # Fixed parameters
    N = 1000                            
    dt = 1./sampling_rate
    tau_exponential_kernel = 5 
    alpha = np.sqrt(1/8)            # SEM * N
    stdq = alpha*mean_firing_rate
    ron = 1./(tau*(1+factor_ron_roff))
    roff = factor_ron_roff*ron
    v_rest = -65
    volt_vec = np.arange(-100, 25, dv).round(3)

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

    # Create qon/qoff
    if qon_qoff_type == 'normal':
        mutheta = 1             #The summed difference between qon and qoff
        alphan = alpha
        regime = 1
        [input_bayes.qon, input_bayes.qoff] = input_bayes.create_qonqoff(mutheta, N, alphan, regime, seed)
    elif qon_qoff_type == 'balanced':
        [input_bayes.qon, input_bayes.qoff] = input_bayes.create_qonqoff_balanced(N, mean_firing_rate, stdq, seed)
    elif qon_qoff_type == 'balanced_uniform':
        minq = 10                  
        maxq = 100
        [input_bayes.qon, input_bayes.qoff] = input_bayes.create_qonqoff_balanced_uniform(N, minq, maxq, seed)
    else: 
        raise SyntaxError('No qon/qoff creation type specified')
    
    #Generate weights and hiddenstate
    input_bayes.get_all()
    input_bayes.x = input_bayes.markov_hiddenstate()

    #Generate exc and inh LUT
    g0_exc, g0_inh = get_g0(v_rest, input_bayes.w)
    g_exc = input_bayes.markov_input(g0_exc)
    g_inh = input_bayes.markov_input(g0_inh)
    
    exc_LUT = get_input_LUT(g_exc, volt_vec, 0)
    inh_LUT = get_input_LUT(g_inh, volt_vec, -75)

    return [exc_LUT, inh_LUT, input_bayes.x]
