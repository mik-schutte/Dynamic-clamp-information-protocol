'''MI.py

File calculating the Mutual information
'''
import numpy as np
from scipy import stats, integrate




#Get hiddenstate
hiddenstate = np.genfromtxt('results/hiddenstate.csv', delimiter=',')
input_theory = np.genfromtxt('results/input_theory.csv', delimiter=',')

#Determine from main.py
tau = 50               
factor_ron_roff = 2
ron = 1./(tau*(1+factor_ron_roff))
roff = factor_ron_roff*ron

p1 = ron/(ron + roff)
H_xx = stats.entropy([p1, (1-p1)], base=2)

#Estimate log-likelihood
theta = 0
def post_log_lik(y, t, ron, roff):
    dydt = ron * (1 + np.exp(-y)) - roff * (1 + np.exp(y))
    return dydt

time = np.arange(0, 10000, 1)

sol = integrate.odeint(post_log_lik, 0, time, args=(ron, roff))

H_xy = np.empty(len(input_theory))
for t in time:
    H_xy[t] = sol[t] + input_theory[t] - theta

    
print(H_xy)