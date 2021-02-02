'''
    input.py
    
    Class containing the input parameters for generating an input current.
    The method is described in the following paper:
    Zeldenrust, F., de Knecht, S., Wadman, W. J., Denève, S., Gutkin, B., Knecht, S. De, Denève, S. (2017). 
    Estimating the Information Extracted by a Single Spiking Neuron from a Continuous Input Time Series. 
    Frontiers in Computational Neuroscience, 11(June), 49. doi:10.3389/FNCOM.2017.00049
    Please cite this reference when using this method.
    
    NB time (dt,T) in ms, freq in Hz, but qon en qoff in MHz
    always first 50 ms silent, then 50 ms noise, then rest   #TODO This isn't the case 
'''
import numpy as np
import matplotlib.pyplot as plt

class Input():
    '''Class containing the input parameters.
    '''
    def __init__(self): 
        # For all
        self.dt = None
        self.T = None          
        self.fHandle = [None, None]
        self.seed = None
        self.input = None

        # For Markov models
        self.ron = None
        self.roff = None
        self.qon = []
        self.qoff = []
        self.kernel = None
        self.kerneltau = None
        self.xseed = None
        self.x = None
        self.xfix = None

    # Get dependend variables
    def get_tvec(self):
        '''Generate tvec and save the length
        '''
        self.tvec = np.arange(self.dt, self.T+self.dt, self.dt)
        self.length = len(self.tvec)

    def generate(self):
        '''Generate input and x from fHandle.
        '''
        if not self.fHandle:
            print('fHandle isn''t provided object')
        else:
            [self.input, self.x] = self.fHandle     

    def get_tau(self):
        '''Generates tau based on the hidden state switch rate
           i.e. ron/roff
        '''
        if self.ron == None or self.roff == None:
            print('Tau not defined, missing ron/roff')
        else:
            self.tau = 1/(self.ron+self.roff)
        
    def get_p0(self):
        '''Generates the probability of finding the hidden state
           in the 'ON' state.
        '''
        if self.ron == None or self.roff == None:
            print('P0 not defined, missing ron/roff')
        else:
            self.p0 = self.ron/(self.ron+self.roff)   

    def get_theta(self):
        '''Generates the firing rate differences.
        '''
        if self.qon == [] or self.qoff == []:
            print('Theta not defined, missing qon/qoff')
        else:
            sum(self.qon-self.qoff)
    
    def get_w(self):
        '''Generates the weight matrix based on qon/qoff.
        '''
        if self.qon == [] or self.qoff == []:
            print('Weight not defined, missing qon/qoff')
        else:
            self.w = np.log(self.qon/self.qoff)

    def get_all(self):
        '''Runs all the functions to create dependent variables.
        '''
        self.get_tvec()
        self.generate()
        self.get_tau()
        self.get_p0()
        self.get_theta()
        self.get_w()


    @staticmethod
    def create_qonqoff(mutheta, N, alphan, regime, qseed=None):
        '''Generates normally distributed [qon, qoff] with qon and qoff 
           being a matrix filled  with the firing rate of each neuron based 
           on the hidden state.
        '''
        np.random.seed(qseed)
        
        qoff = np.random.randn(N, 1) 
        qon = np.random.randn(N, 1)
        if N > 1:
            #Creates a q distribution with a standard deviation of 1 
            qoff = qoff/np.std(qoff)
            qon = qon/np.std(qon)
        qoff = qoff - np.mean(qoff)
        qon = qon - np.mean(qon)

        if regime == 1:   
            #Coincedence regime !! No E/I balance, little negative weights
            qoff = (alphan*qoff+1)*mutheta/N
            qon = (alphan*qon+2)*mutheta/N
        else:
            #Push-pull regime !! E/I balance, negative weights
            qoff = (alphan*qoff+1)*mutheta/np.sqrt(N)
            qon = (alphan*qon+1+1/np.sqrt(N))*mutheta/np.sqrt(N)
        
        #Set all negative firing rates to 0
        qoff[qoff<0] = abs(qoff[qoff<0])
        qon[qon<0] = abs(qon[qon<0])
        return [qon, qoff]
    

    @staticmethod
    def create_qonqoff_balanced(N,  meanq, stdq, qseed=None):
        '''Generates normally distributed [qon, qoff] with qon and qoff 
           being a matrix filled  with the firing rate of each neuron based 
           on the hidden state.
        '''
        np.random.seed(qseed)

        qoff = np.random.randn(N, 1)
        qon = np.random.randn(N, 1)
        if N > 1: 
            qoff = qoff/np.std(qoff)
            qon = qon/np.std(qon)

        qoff = stdq*(qoff-np.mean(qoff))+meanq
        qon = stdq*(qon-np.mean(qon))+meanq

        qoff[qoff<0] = abs(qoff[qoff<0])
        qon[qon<0] = abs(qon[qon<0])
        return [qon, qoff]


    @staticmethod
    def create_qonqoff_balanced_uniform(N, minq, maxq, qseed=None):
        '''Generates uniformly distributed [qon, qoff] with qon and qoff 
           being a matrix filled with the firing rate of each neuron based 
           on the hidden state.
        '''
        np.random.seed(qseed)
        
        qoff = np.random.rand(N, 1)
        qoff = minq + np.multiply((maxq-minq), qoff)
        qon = np.random.rand(N, 1)
        qon = minq + np.multiply((maxq-minq), qon)
        return [qon, qoff]


    def markov_hiddenstate(self): 
        '''Takes ron and roff from class object and generates
           the hiddenstate if xfix is empty.
        '''
        np.random.seed(self.xseed)
        
        # Generate x
        if self.xfix == None:
            self.get_p0()
            xs = np.zeros(np.shape(self.tvec)) 

            #Initial value 
            i = np.random.rand()
            if i < self.p0:
                xs[0] = 1
            else:
                xs[0] = 0

            # Make x
            for n in np.arange(1, self.length): 
                i = np.random.rand()
                if xs[n-1] == 1: 
                    if i < self.roff*self.dt:
                        xs[n] = 0
                    else:
                        xs[n] = 1
                else: 
                    if i < self.ron*self.dt:
                        xs[n] = 1
                    else:
                        xs[n] = 0
        else:
            xs = self.xfix

        return xs


    def markov_input(self, dynamic=False):
        '''Takes qon, qoff and hiddenstate and generates input.
           Optionally when dynamic is a dictinary of g0_values it
           generates a conductance over time based on the hidden state. 
        '''
        xs = self.x
        nt = self.length 
        w = np.log(self.qon/self.qoff) 

        if dynamic:
            ni = dynamic.keys()
        else:
            ni = range(len(self.qon))

        # Make spike trains (implicit)
        stsum = np.zeros((nt, 1))
        if self.kernel != None:
            if self.kernel == 'exponential':
                tfilt = np.arange(0, 5*self.kerneltau+self.dt, self.dt)
                kernelf = np.exp(-tfilt/self.kerneltau)
                kernelf = kernelf/(self.dt*sum(kernelf)) 
            elif self.kernel == 'delta':
                kernelf = 1./self.dt
        
        xon = np.where(xs==1)
        xoff = np.where(xs==0)
        np.random.seed(self.seed)
        
        # Create the input generated by the artificial neural network
        for k in ni:
            randon = np.random.rand(np.shape(xon)[0],np.shape(xon)[1])
            randoff = np.random.rand(np.shape(xoff)[0], np.shape(xoff)[1])
            sttemp = np.zeros((nt, 1))
            sttempon = np.zeros(np.shape(xon))
            sttempoff = np.zeros(np.shape(xoff))

            sttempon[randon < self.qon[k]*self.dt] = 1.
            sttempoff[randoff < self.qoff[k]*self.dt] = 1.
            
            sttemp[xon] = np.transpose(sttempon)
            sttemp[xoff] = np.transpose(sttempoff)

            if dynamic:
                stsum = stsum + dynamic[k]*sttemp
            else:
                stsum = stsum + w[k]*sttemp 

            # #SanityCheck for individual spikes
            # plt.plot(sttemp)
            # plt.show()

        if self.kernel != None:
            stsum = np.convolve(stsum.flatten(), kernelf, mode='full')

        stsum = stsum[0:nt]
        ip = stsum 
        return ip