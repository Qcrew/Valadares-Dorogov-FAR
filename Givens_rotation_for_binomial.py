#%%Simulation to obtain number-splitting
import matplotlib.pyplot as plt
import numpy as np
from qutip import *
from scipy.optimize import minimize
import h5py
import scipy as sc
import time
start_time = time.time()#checking how long the code takes

cdim = 6

X02 = basis(cdim,0)*basis(cdim,2).dag()+basis(cdim,2)*basis(cdim,0).dag()
X04 = basis(cdim,0)*basis(cdim,4).dag()+basis(cdim,4)*basis(cdim,0).dag()
X24 = basis(cdim,2)*basis(cdim,4).dag()+basis(cdim,4)*basis(cdim,2).dag()

theta = -0.3248
U1 = (-1j*theta/(2*np.sqrt(2))*(X02+X24)).expm()

phi1 = 2*np.arctan(np.tan(theta/4)/np.sqrt(2))
phi2 = 4*np.arcsin(np.sin(theta/4)/np.sqrt(2))


U2 = (-1j*(phi1/2)*(X02)).expm()*(-1j*phi2/2*(X24)).expm()*(-1j*(phi1/2)*(X02)).expm()

t = np.allclose(U1.full(), U2.full())

print(t)

#%%
U1
#%%
U2
#%%

print("")
print("--- %s seconds ---" % (time.time() - start_time))
#%%

