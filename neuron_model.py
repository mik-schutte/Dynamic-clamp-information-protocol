'''
neuron_model.py

File containing the HH neuron model in Brian2 as used in the Wang & Buszaki (1996) article.
From: https://brian2.readthedocs.io/en/stable/examples/frompapers.Wang_Buszaki_1996.html
'''
from code.input import Input
from code.make_input_experiments import make_input_experiments
from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

start_scope()
defaultclock.dt = 0.01*ms

Cm = 1*uF # /cm**2
Iapp = I_inject
gL = 0.1*msiemens
EL = -65*mV
ENa = 55*mV
EK = -90*mV
gNa = 35*msiemens
gK = 9*msiemens
gACh = 45*psiemens #for a single channel
EACh = 0*mV

eqs = '''
dv/dt = (-gNa*m**3*h*(v-ENa)-gK*n**4*(v-EK)-gL*(v-EL) + Iapp(t,i) + Idynamic)/Cm : volt

alpha_m = 0.1/mV*10.*mV/exp(-(v+35.*mV)/(10.*mV))/ms : Hz
alpha_h = 0.07*exp(-(v+58.*mV)/(20.*mV))/ms : Hz
alpha_n = 0.01/mV*10.*mV/exp(-(v+34.*mV)/(10.*mV))/ms : Hz

beta_m = 4.*exp(-(v+60.*mV)/(18.*mV))/ms : Hz
beta_h = 1./(exp(-0.1/mV*(v+28.*mV))+1)/ms : Hz
beta_n = 0.125*exp(-(v+44.*mV)/(80.*mV))/ms : Hz

m = alpha_m/(alpha_m+beta_m) : 1
dh/dt = 5.*(alpha_h*(1-h)-beta_h*h) : 1
dn/dt = 5.*(alpha_n*(1-n)-beta_n*n) : 1

dynamic = 1/ (v/Iapp(t,i)) : siemens
Idynamic : amp
'''

neuron = NeuronGroup(1, eqs, method='exponential_euler', threshold = 'v>-20*mV')
neuron.v = -70*mV
neuron.h = 1
M = StateMonitor(neuron, ['v', 'dynamic'], record=0)
S = SpikeMonitor(neuron)

print('Running...')
run(2000*ms, report='text')
print('Done')

plot_voltage_and_current_traces(M, I_inject, 'Wang_Buszaki')