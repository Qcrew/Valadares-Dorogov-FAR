#%% Simulation for X, Y, and Z logical rotations on binomial encoding
import numpy as np
from qutip import *

cdim = 6
theta = -0.3248

#%% 1. Define Basic Operators ---
X02 = basis(cdim,0)*basis(cdim,2).dag() + basis(cdim,2)*basis(cdim,0).dag()
X04 = basis(cdim,0)*basis(cdim,4).dag() + basis(cdim,4)*basis(cdim,0).dag()
X24 = basis(cdim,2)*basis(cdim,4).dag() + basis(cdim,4)*basis(cdim,2).dag()

Y02 = -1j*basis(cdim,0)*basis(cdim,2).dag() + 1j*basis(cdim,2)*basis(cdim,0).dag()
Y24 = -1j*basis(cdim,2)*basis(cdim,4).dag() + 1j*basis(cdim,4)*basis(cdim,2).dag()

#%% 2. Define Logical Basis and Target Operators ---
L0 = basis(cdim, 2)
L1 = (basis(cdim, 0) + basis(cdim, 4)) / np.sqrt(2)

X_L = (X02 + X24) / np.sqrt(2)
Y_L = (Y24 - Y02) / np.sqrt(2)
Z_L = L0 * L0.dag() - L1 * L1.dag()

#%% 3. Verify R_x ---
phi1 = 2*np.arctan(np.tan(theta/4)/np.sqrt(2))
phi2 = 4*np.arcsin(np.sin(theta/4)/np.sqrt(2))

Ux_target = (-1j * theta / 2 * X_L).expm()
Ux_decomp = (-1j * (phi1/2) * X02).expm() * (-1j * (phi2/2) * X24).expm() * (-1j * (phi1/2) * X02).expm()

print(f"R_x exact matrix match: {np.allclose(Ux_target.full(), Ux_decomp.full())}")
Ux_target
#%%
Ux_decomp

#%% 4. Verify R_y ---
Uy_target = (-1j * theta / 2 * Y_L).expm()

Uy_decomp = (1j * (phi1/2) * Y02).expm() * (-1j * (phi2/2) * Y24).expm() * (1j * (phi1/2) * Y02).expm()
print(f"R_y exact matrix match: {np.allclose(Uy_target.full(), Uy_decomp.full())}")
Uy_target
#%%
Uy_decomp
#%% 5. Verify R_z ---
Uz_target = (-1j * theta / 2 * Z_L).expm()

# Decomposition: R04(-2*theta) = exp(i * theta * X04)
Uz_decomp = (1j * theta * X04).expm()

# Uz_target applies exp(-i*theta/2) to L0 and exp(i*theta/2) to L1.
# Uz_decomp applies 1 to L0 and exp(i*theta) to L1.
# We must factor out the global phase difference to compare them.
global_phase = np.exp(-1j * theta / 2)
z_match_L0 = np.allclose((Uz_target * L0).full(), (global_phase * Uz_decomp * L0).full())
z_match_L1 = np.allclose((Uz_target * L1).full(), (global_phase * Uz_decomp * L1).full())
print(f"R_z logical state match (up to global phase): {z_match_L0 and z_match_L1}")
Uz_target
#%%
Uz_decomp