import numpy as np
from brian2 import *
from code.make_dynamic_experiments import make_dynamic_experiments

'''Docstring
'''

# Set parameters
baseline = 0           
amplitude_scaling = 1     
tau = 50               
factor_ron_roff = 2    
mean_firing_rate = (0.5)/1000 
sampling_rate = 5      
dt = 1/sampling_rate 
dv = 0.5
duration = 2000
qon_qoff_type = 'balanced'
input_type = 'dynamic'
Er_exc, Er_inh = (0, -75)

#Generate input and hiddenstate
exc_LUT, inh_LUT, input_theory, hiddenstate = make_dynamic_experiments(qon_qoff_type, baseline, amplitude_scaling, tau, factor_ron_roff, mean_firing_rate, sampling_rate, duration, dv)

#Get conductances...
g_exc = abs(exc_LUT[-65] / (-65 - Er_exc))*mS 
g_inh = abs(inh_LUT[-65] / (-65 - Er_inh))*mS 
g_exc = TimedArray(g_exc, dt=0.2*ms)
g_inh = TimedArray(g_inh, dt=0.2*ms)
#and input ready for brian2
input_currentx = baseline + input_theory*amplitude_scaling
input_current = input_currentx*uamp
input_current = TimedArray(input_current, dt=0.2*ms)

#Make Neuron model
start_scope()
defaultclock.dt = 0.01*ms

Cm = 1*uF # /cm**2
gL = 0.1*msiemens
EL = -65*mV
ENa = 55*mV
EK = -90*mV
E_exc = 0*mV
E_inh = -75*mV
gNa = 35*msiemens
gK = 9*msiemens

eqs_dynamic = '''I_exc = g_exc(t) * (E_exc - v) : amp
                 I_inh = g_inh(t) * (E_inh - v) : amp
                 I_inj = I_exc + I_inh : amp'''

eqs_current = '''I_inj = input_current(t) : amp'''

eqs = '''
dv/dt = (gNa * m**3 * h * (ENa - v) + gK * n**4 * (EK - v) + gL * (EL - v) + I_inj)/Cm : volt

alpha_m = 0.1/mV * 10.*mV / exp(-(v + 35.*mV) / (10.*mV))/ms : Hz
alpha_h = 0.07 * exp(-(v + 58.*mV) / (20.*mV))/ms : Hz
alpha_n = 0.01/mV * 10.*mV / exp(-(v+34.*mV) / (10.*mV))/ms : Hz

beta_m = 4. * exp(-(v+60.*mV) / (18.*mV))/ms : Hz
beta_h = 1. / (exp(-0.1/mV * (v + 28.*mV)) + 1)/ms : Hz
beta_n = 0.125 * exp(-(v + 44.*mV) / (80.*mV))/ms : Hz

m = alpha_m / (alpha_m + beta_m) : 1
dh/dt = 5. * (alpha_h * (1 - h) - beta_h * h) : 1
dn/dt = 5. * (alpha_n * (1 - n) - beta_n * n) : 1
'''
current_neuron = NeuronGroup(1, eqs+eqs_current, method='exponential_euler', threshold='v>-20*mV')
current_neuron.v = -65*mV
current_neuron.h = 1
current_M = StateMonitor(current_neuron, ['v', 'I_inj'], record=0)

dynamic_neuron = NeuronGroup(1, eqs+eqs_dynamic, method='exponential_euler', threshold ='v>-20*mV')
dynamic_neuron.v = -65*mV
dynamic_neuron.h = 1
dynamic_M = StateMonitor(dynamic_neuron, ['v', 'I_inj', 'I_exc', 'I_inh'], record=0)

#Run
run(2000*ms, report='text')

#Calculate Mutual Information