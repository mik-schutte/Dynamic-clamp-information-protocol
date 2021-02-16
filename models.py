''' 
    models.py

    Python file containing different neuron models used in simulations.
'''
import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np

def simulate_Wang_Buszaki(inj_input, simulation_time, clamp_type='current'):
    ''' Hodgkin-Huxley model of a hippocampal (CA1) interneuron.

        INPUT:
            inj_input ((Tuple of) TimedArray): Input current or conductances (g_exc, g_inh)
            duration (float): Simulation time [milliseconds]
            clamp_type (string): type of input, ['current' or 'dynamic'] default = current

        OUTPUT:
            StateMonitor: Brian2 StateMonitor with recorded fields
            ['v', 'input' or 'conductance']

        Xiao-Jing Wang & György Buzsáki, (1996). Gamma Oscillation 
        by synaptic inhibition in a hippocampal interneuronal network 
        model. doi: https://doi.org/10.1523/JNEUROSCI.16-20-06402.1996 

        Original from: https://brian2.readthedocs.io/en/stable/examples/frompapers.Wang_Buszaki_1996.html   
    '''

    # Determine the simulation
    if clamp_type == 'current':
        eqs_input = '''I_inj = inj_input(t) : amp'''
        tracking = ['v', 'I_inj']

    elif clamp_type =='dynamic':
        g_exc, g_inh = inj_input
        eqs_input = '''I_exc = g_exc(t) * (0*mV - v) : amp
                 I_inh = g_inh(t) * (-75*mV - v) : amp
                 I_inj = I_exc + I_inh : amp'''
        tracking = ['v', 'g_exc', 'g_inh']

    # Neuron parameters
    Cm = 1 * b2.uF # /cm**2
    gL = 0.1*b2.msiemens
    gNa = 35*b2.msiemens
    gK = 9*b2.msiemens
    EL = -65*b2.mV
    ENa = 55*b2.mV
    EK = -90*b2.mV
    
    # Model the neuron with differential equations
    eqs = '''
        dv/dt = (gNa * m**3 * h * (ENa - v) + gK * n**4 * (EK - v) + gL * (EL - v) + I_inj)/Cm : volt

        alpha_m = 0.1/mV * 10.*mV / exp(-(v + 35.*mV) / (10.*mV))/ms : Hz
        alpha_h = 0.07 * exp(-(v + 58.*mV) / (20.*mV))/ms : Hz
        alpha_n = 0.01/mV * 10.*mV / exp(-(v + 34.*mV) / (10.*mV))/ms : Hz

        beta_m = 4. * exp(-(v + 60.*mV) / (18.*mV))/ms : Hz
        beta_h = 1. / (exp(-0.1/mV * (v + 28.*mV)) + 1)/ms : Hz
        beta_n = 0.125 * exp(-(v + 44.*mV) / (80.*mV))/ms : Hz

        m = alpha_m / (alpha_m + beta_m) : 1
        dh/dt = 5. * (alpha_h * (1 - h) - beta_h * h) : 1
        dn/dt = 5. * (alpha_n * (1 - n) - beta_n * n) : 1
        '''

    # Neuron & parameter initialization
    neuron = b2.NeuronGroup(1, eqs+eqs_input, method='exponential_euler')
    neuron.v = -70*b2.mV
    neuron.h = 1

    # Track the parameters during simulation
    M = b2.StateMonitor(neuron, tracking, record=True)

    # Run the simulation
    net = b2.Network(neuron)
    net.add(M)
    net.run(simulation_time, report='text')

    return M


def simulate_barrel_PC(inj_input, simulation_time, clamp_type):
    ''' Hodgkin-Huxley model of a Pyramidal Cell in the rat barrel cortex.

        INPUT:
            inj_input ((Tuple of) TimedArray): Input current or conductances (g_exc, g_inh)
            duration (float): Simulation time [milliseconds]
            clamp_type (string): type of input, ['current' or 'dynamic'] default = current

        OUTPUT:
            StateMonitor: Brian2 StateMonitor with recorded fields
            ['v', 'input' or 'conductance']

        The parameters used in this model have been fitted by Xenia Sterl under 
        the supervision of Fleur Zeldenrust. Full description can be found at:
        Xenia Sterl, Fleur Zeldenrust, (2020). Dopamine modulates firing rates and information
        transfer in inhibitory and excitatory neurons of rat barrel cortex, but shows no clear
        influence on neuronal parameters. (Unpublished bachelor's thesis)
    '''

    # Determine the simulation
    if clamp_type == 'current':
        eqs_input = '''I_inj = inj_input(t) : amp'''
        tracking = ['v', 'I_inj']

    elif clamp_type =='dynamic':
        g_exc, g_inh = inj_input
        eqs_input = '''I_exc = g_exc(t) * (0*mV - v) : amp
                 I_inh = g_inh(t) * (-75*mV - v) : amp
                 I_inj = I_exc + I_inh : amp'''
        tracking = ['v', 'g_exc', 'g_inh']

    # Neuron parameters
    area = 20000*b2.umetre**2
    Cm = 1.6287344247018984e-10*b2.farad/area * b2.cm**2 
    gL = 9.745664712129547e-09*b2.siemens/area * b2.cm**2 
    gNa = 0.00011423100873563258*b2.siemens/area * b2.cm**2 
    gK = 2.4881444790241606e-05*b2.siemens/area * b2.cm**2 
    EL = -65*b2.mV
    ENa = 50*b2.mV
    EK = -90*b2.mV
    k_m = 0.004400673179014958*b2.volt
    k_h = 0.008439593401769845*b2.volt
    Vh_h = -0.02973884050950356*b2.volt
    VT = -63*b2.mV
    
    # Model the neuron with differential equations
    eqs = '''
        Vh_m = 3.583881 * k_m - 53.294454*mV : volt
        m = 1 / (1 + exp(-(v - Vh_m) / k_m)) : 1
        h = 1 / (1 + exp((v - Vh_h) / k_h)) : 1

        alpha_n = (0.032 * 5. / exprel((15. -v/mV + VT/mV) / 5.))/ms : Hz
        beta_n = (0.5 * exp((10. - v/mV + VT/mV) / 40.))/ms : Hz
        dn/dt = alpha_n * (1 - n) - beta_n * n : 1

        I_leak = gL * (v - EL) : amp
        I_Na = gNa * m**3 * h * (v - ENa) : amp
        I_K = gK * n**4 * (v - EK) : amp

        dv/dt = (-(I_leak + I_Na + I_K) + I_inj) / Cm : volt
        '''    

    # Neuron & parameter initialization
    neuron = b2.NeuronGroup(1, model=eqs+eqs_input, method='exponential_euler',
                        threshold ='m > 0.5', refractory=2*b2.ms, reset=None, dt=0.5*b2.ms)
    neuron.v = -65*b2.mV

    # Track the parameters during simulation
    M = b2.StateMonitor(neuron, tracking, record=True)

    # Run the simulation
    net = b2.Network(neuron)
    net.add(M)
    net.run(simulation_time, report='text')

    return M


inj_input = np.genfromtxt('results/input_current.csv')
inj_input = b2.TimedArray(inj_input*b2.uamp, dt=0.2*b2.ms)
simulation_time = 2000*b2.ms

monitor = simulate_barrel_PC(inj_input, simulation_time, clamp_type='current')
print(monitor)