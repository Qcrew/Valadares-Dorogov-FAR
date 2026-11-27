#%%
import numpy as np
from qutip import *
from scipy.optimize import minimize
import scipy.optimize as sco
import time
from f1_path_utility import *

#solution 600
# x = np.array([
#     -0.449767, -0.774022, 5.437919, -3.550535, 6.198303, -4.473629, 2.863675, -0.948960, -6.751871, 5.916508, 3.571885, -0.715351, 1.655844, -0.446487, -0.796172, 0.830570, -4.945060, 3.109441, 6.512144, 1.213028
# ])
#solution 800
x = np.array([
    -0.709250, 0.750886, 1.835224, 1.422399, 2.518140, -0.422738, 3.136474, -1.947848, -4.838214, 0.514579, 2.918720, 2.752379, 0.675974, 1.933002, -1.283012, 1.088491, 2.801328, -3.910986, 0.726165, 1.458392
])

# Constants
g = 2 * np.pi * 12.07e-3  # GHz
wq = 2 * np.pi * (6900 - 30.5)*1e-3  # GHz
wr = wq

# Determining the path and frequencies from the input transition
if __name__ == "__main__":
    transition_input = "0mto5p"
    init, final = parse_transition_string(transition_input)

    path = generate_path(init, final)
    print("Path taken is:")
    print(path)

    wns = np.array(get_transition_freqs(path, g, wq, wr))
    np.set_printoptions(legacy="1.25")
    print("Frequencies (in MHz) are:")
    print(wns)

# ----------------------------
# Simulation Parameters
# ----------------------------
Tmax = 800  # total duration in ns
AD = True
TAD = 200 # for adiabatic detuning
# newparameter = True  # whether to start with a fresh seed
theta, N = np.pi / 4, len(wns)  # target state parameters
M = 2  # number of components for An, Bn

# ----------------------------
# System Setup
# ----------------------------
qdim, cdim = 2, 10

q, c = destroy(qdim), destroy(cdim)
Q, C = tensor(q, qeye(cdim)), tensor(qeye(qdim), c)
Qd, Cd = Q.dag(), C.dag()

H0 = g * (C * Qd + Cd * Q)

# T1 = np.inf#100e3
# # T2 = 50e3
# Tphi = np.inf#1 / (1 / T2 - 0.5 / T1)
# cavT1 = np.inf

T1 = 50e3# our exp 10e3, 15e3
T2 = 30e3# our exp 1.3e3, 2.8e3
Tphi = 1 / (1 / T2 - 0.5 / T1)
cavT1 = 0.5e6

c_ops = [
np.sqrt(1 / T1) * Q,  # Qubit Relaxation
np.sqrt(2 / Tphi) * Qd * Q,  # Qubit Dephasing
np.sqrt(1 / cavT1) * C,  # Cavity Relaxation
]

tlist = np.linspace(0, Tmax, 100)
T = tlist[-1]
tlist_AD = np.linspace(0, TAD, 100)

N0 = init[0]
if init == (0, 0):
    psi0 = tensor(fock(qdim, 0), fock(cdim, 0))
else: 
    psi0 = (np.cos(theta) * tensor(fock(qdim, 0), fock(cdim, N0)) + 
          (-1)**(init[1]+1) * np.sin(theta) * tensor(fock(qdim, 1), fock(cdim, N0 - 1)))  

Nf = final[0]
if final == (0, 0):
    psitar = tensor(fock(qdim, 0), fock(cdim, 0))
else: 
    psitar = (np.cos(theta) * tensor(fock(qdim, 0), fock(cdim, Nf)) + 
          (-1)**(final[1]+1) * np.sin(theta) * tensor(fock(qdim, 1), fock(cdim, Nf - 1)))    

# ----------------------------
# Control Envelopes
# ----------------------------
def An(t, an):
    base = 2 * np.pi * t / T
    return sum(an[k] * (1 - np.cos((k + 1) * base)) for k in range(M))

def Bn(t, bn):
    base = 2 * np.pi * t / T
    return sum(bn[k] * np.sin((k + 1) * base) for k in range(M))

def build_coeffs(wnq, Man, Mbn):
    def H1_coeff(t, args):
        return sum(0.5 * (An(t, Man[n]) + 1j * Bn(t, Mbn[n])) * np.exp(-1j * wnq[n] * t) for n in range(N))
    
    def H2_coeff(t, args):
        return sum(0.5 * (An(t, Man[n]) - 1j * Bn(t, Mbn[n])) * np.exp(1j * wnq[n] * t) for n in range(N))
    
    return H1_coeff, H2_coeff


# freqs_init = compute_freqs()
freqs_rad = wns * 2 * np.pi * 1e-3  # rad/GHz

# ----------------------------
# Initial Guess for Coefficients
# ----------------------------
def scaled_random(base, scale_type='positive'):
    if scale_type == 'positive':
        return base * (0.5 + np.random.rand(*base.shape))
    elif scale_type == 'symmetric':
        return base * (2 * np.random.rand(*base.shape) - 1)


# ----------------------------
# Cost Function (1 - Fidelity)
# ----------------------------
def cost(x, freqs=freqs_rad,AD=True):
    Man = x[:N*M].reshape([N, M]) * 2 * np.pi * 1e-3
    Mbn = x[N*M:].reshape([N, M]) * 2 * np.pi * 1e-3

    H1 = tensor(create(qdim), qeye(cdim))
    H2 = tensor(destroy(qdim), qeye(cdim))
    H1_coeff, H2_coeff = build_coeffs(freqs - wq, Man, Mbn)#freqs - wq because of RWA

    H = [H0, [H1, H1_coeff], [H2, H2_coeff]]
    result = mesolve(H, ket2dm(psi0), tlist, c_ops=c_ops)#, options=Options(nsteps=10000))

    if AD == True:
        result = mesolve(H0, result.states[-1], tlist_AD, c_ops=c_ops)#, options=Options(nsteps=10000))

    ff = fidelity(ket2dm(psitar), result.states[-1])**2
    # print(ff)
    return 1 - ff, result.states[-1]

fidelity  = 1 - cost(x, freqs_rad,AD)[0]
# uout = cost(x, freqs_rad)[1]
print(f'fidel is {fidelity}')

# ute = psitar
# ute = (psi0 + 1j*psitar).unit()
# print(f'fidel to superpos is {np.abs(ute.dag()*uout)**2}')