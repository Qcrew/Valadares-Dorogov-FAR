#%%
import numpy as np
from qutip import *

# ============================================================
# IDEAL GATE
# ============================================================

Rid = (-1j * sigmax() * (-np.pi/4) / 2).expm().full()

U_target = Rid


# ============================================================
# DATA
# ============================================================

data0 = np.load(r"C:\Users\tanju\Dropbox\NTU Grad\Research\Python codes\nus_tomo_23\20250714_Allcoherent\All_Resonant_Simulations\data\states\r0.npz")
data2 = np.load(r"C:\Users\tanju\Dropbox\NTU Grad\Research\Python codes\nus_tomo_23\20250714_Allcoherent\All_Resonant_Simulations\data\states\r2.npz")
data4 = np.load(r"C:\Users\tanju\Dropbox\NTU Grad\Research\Python codes\nus_tomo_23\20250714_Allcoherent\All_Resonant_Simulations\data\states\r4.npz")
data6 = np.load(r"C:\Users\tanju\Dropbox\NTU Grad\Research\Python codes\nus_tomo_23\20250714_Allcoherent\All_Resonant_Simulations\data\states\r6.npz")

data1 = np.load(r"C:\Users\tanju\Dropbox\NTU Grad\Research\Python codes\nus_tomo_23\20250714_Allcoherent\All_Resonant_Simulations\data\states\r1.npz")
data3 = np.load(r"C:\Users\tanju\Dropbox\NTU Grad\Research\Python codes\nus_tomo_23\20250714_Allcoherent\All_Resonant_Simulations\data\states\r3.npz")
data5 = np.load(r"C:\Users\tanju\Dropbox\NTU Grad\Research\Python codes\nus_tomo_23\20250714_Allcoherent\All_Resonant_Simulations\data\states\r5.npz")
data7 = np.load(r"C:\Users\tanju\Dropbox\NTU Grad\Research\Python codes\nus_tomo_23\20250714_Allcoherent\All_Resonant_Simulations\data\states\r7.npz")

# data1 = np.load(r"C:\Users\tanju\Dropbox\NTU Grad\Research\Python codes\nus_tomo_23\20250714_Allcoherent\All_Resonant_Simulations\data\states\r1BC.npz")
# data3 = np.load(r"C:\Users\tanju\Dropbox\NTU Grad\Research\Python codes\nus_tomo_23\20250714_Allcoherent\All_Resonant_Simulations\data\states\r3BC.npz")
# data5 = np.load(r"C:\Users\tanju\Dropbox\NTU Grad\Research\Python codes\nus_tomo_23\20250714_Allcoherent\All_Resonant_Simulations\data\states\r5BC.npz")
# data7 = np.load(r"C:\Users\tanju\Dropbox\NTU Grad\Research\Python codes\nus_tomo_23\20250714_Allcoherent\All_Resonant_Simulations\data\states\r7BC.npz")



# ============================================================
# COLLECT DATA
# ============================================================

R = [
    data0['rho_data'],
    data1['rho_data'],
    data2['rho_data'],
    data3['rho_data'],
    data4['rho_data'],
    data5['rho_data'],
    data6['rho_data'],
    data7['rho_data']
]


# ============================================================
# EXTRACT 2x2 QUBIT DENSITY MATRIX
# ============================================================

def qq(R):

    r = np.zeros((2, 2), dtype=complex)

    r[0, 0] = R[1, 1]
    r[0, 1] = R[1, 4]
    r[1, 0] = R[4, 1]
    r[1, 1] = R[4, 4]

    return r


# ============================================================
# NORMALIZE DENSITY MATRICES
# ============================================================

def normalize(rho):

    rho = np.asarray(rho, dtype=complex)

    tr = np.trace(rho)

    if abs(tr) < 1e-14:
        raise ValueError("Density matrix has zero trace.")

    return rho #/ tr


# ============================================================
# INPUT STATES
# ============================================================

input_states = [
    normalize(qq(R[0])),
    normalize(qq(R[2])),
    normalize(qq(R[4])),
    normalize(qq(R[6]))
]


# ============================================================
# OUTPUT STATES
# ============================================================

output_states = [
    normalize(qq(R[1])),
    normalize(qq(R[3])),
    normalize(qq(R[5])),
    normalize(qq(R[7]))
]


# ============================================================
# PRINT TRACES
# ============================================================

print("\n========================================")
print("INPUT / OUTPUT STATE TRACES")
print("========================================")

for i in range(4):

    print(
        f"State {i}: "
        f"Tr(input) = {np.trace(input_states[i]):.10f}, "
        f"Tr(output) = {np.trace(output_states[i]):.10f}"
    )


# ============================================================
# IDEAL OUTPUT STATES
# ============================================================

ideal_output_states = [
    U_target @ rho @ U_target.conj().T
    for rho in input_states
]


# ============================================================
# INDIVIDUAL STATE FIDELITIES
# ============================================================

F_state = np.zeros(4)

print("\n========================================")
print("INDIVIDUAL STATE FIDELITIES")
print("========================================")

for i in range(4):

    # QuTiP fidelity() returns sqrt(F) for the usual
    # density-matrix fidelity convention.
    #
    # Therefore square it to obtain F.

    F_state[i] = fidelity(
        Qobj(ideal_output_states[i]),
        Qobj(output_states[i])
    )**2

    print(
        f"State {i}: "
        f"{F_state[i]:.10f}"
    )

print(
    f"\nMean state fidelity = "
    f"{np.mean(F_state):.10f}"
)


# ============================================================
# VECTORIZE MATRIX
#
# NumPy default = row-major
#
# [[a,b],
#  [c,d]]
#
# -> [a,b,c,d]
# ============================================================

def vec(rho):

    return np.asarray(rho, dtype=complex).reshape(4)


# ============================================================
# CONSTRUCT X AND Y
# ============================================================

X = np.column_stack([
    vec(rho) for rho in input_states
])

Y = np.column_stack([
    vec(rho) for rho in output_states
])


# ============================================================
# CHECK LINEAR INDEPENDENCE
# ============================================================

rank = np.linalg.matrix_rank(X)

condition_number = np.linalg.cond(X)

print("\n========================================")
print("PROCESS TOMOGRAPHY")
print("========================================")

print(f"Rank of X          = {rank}")
print(f"Condition number X = {condition_number:.6e}")

if rank < 4:

    raise ValueError(
        "The four input states are not linearly independent."
    )


# ============================================================
# RECONSTRUCT SUPEROPERATOR
#
# vec(E(rho)) = S @ vec(rho)
#
# ============================================================

S = Y @ np.linalg.inv(X)


# ============================================================
# CHANNEL
# ============================================================

def channel(rho):

    rho_out = S @ vec(rho)

    return rho_out.reshape(2, 2)


# ============================================================
# CHECK THAT THE RECONSTRUCTED CHANNEL
# REPRODUCES THE FOUR MEASURED OUTPUT STATES
# ============================================================

print("\n========================================")
print("CHANNEL RECONSTRUCTION CHECK")
print("========================================")

for i in range(4):

    reconstructed_output = channel(input_states[i])

    F_reconstruction = fidelity(
        Qobj(reconstructed_output),
        Qobj(output_states[i])
    )**2

    print(
        f"State {i}: "
        f"F(reconstructed, measured) = "
        f"{F_reconstruction:.10f}"
    )


# ============================================================
# CONSTRUCT EXPERIMENTAL CHOI STATE
#
# rho_choi_exp =
#
# (I tensor E)(|Phi><Phi|)
#
# where
#
# |Phi> = (|00> + |11>) / sqrt(2)
#
# The INPUT maximally entangled state is pure.
#
# The resulting Choi state of E can be MIXED.
# ============================================================

phi = np.array([
    [1],
    [0],
    [0],
    [1]
], dtype=complex) / np.sqrt(2)


rho_phi = phi @ phi.conj().T


# ============================================================
# EXPERIMENTAL CHOI MATRIX
# ============================================================

choi_exp = np.zeros((4, 4), dtype=complex)


for i in range(2):

    for j in range(2):

        basis = np.zeros((2, 2), dtype=complex)

        basis[i, j] = 1.0

        E_basis = channel(basis)

        choi_exp += np.kron(
            basis,
            E_basis
        )


# Normalize Choi state
choi_exp = choi_exp / 2.0


# Remove numerical non-Hermiticity
choi_exp = (
    choi_exp +
    choi_exp.conj().T
) / 2.0


# ============================================================
# IDEAL CHOI STATE
#
# For a unitary ideal gate, this Choi state is pure.
# ============================================================

phi_ideal = (
    np.kron(np.eye(2), U_target)
    @ phi
)


choi_ideal = (
    phi_ideal @
    phi_ideal.conj().T
)


# Normalize explicitly
choi_ideal = choi_ideal / np.trace(choi_ideal)


# ============================================================
# CHOI TRACES
# ============================================================

print("\n========================================")
print("CHOI STATE CHECK")
print("========================================")

print(
    "Tr(Choi experimental) = ",
    np.trace(choi_exp)
)

print(
    "Tr(Choi ideal)        = ",
    np.trace(choi_ideal)
)


# ============================================================
# CHOI EIGENVALUES
# ============================================================

choi_exp_eigenvalues = np.linalg.eigvalsh(
    choi_exp
)

choi_ideal_eigenvalues = np.linalg.eigvalsh(
    choi_ideal
)


print("\nExperimental Choi eigenvalues:")
print(choi_exp_eigenvalues)

print("\nIdeal Choi eigenvalues:")
print(choi_ideal_eigenvalues)

print(
    "\nMinimum experimental Choi eigenvalue = ",
    np.min(choi_exp_eigenvalues)
)


# ============================================================
# MIXED-STATE PROCESS FIDELITY
#
# Uhlmann fidelity:
#
# F(rho,sigma) =
#
# [ Tr sqrt(
#       sqrt(rho) sigma sqrt(rho)
#   ) ]^2
#
# QuTiP fidelity() returns the square-root form,
# therefore we square it.
#
# ============================================================

F_process = fidelity(
    Qobj(choi_ideal),
    Qobj(choi_exp)
)**2


F_process = np.real(F_process)

F_process = np.clip(
    F_process,
    0.0,
    1.0
)


# ============================================================
# AVERAGE GATE FIDELITY
#
# For a trace-preserving channel:
#
# F_avg = (d F_process + 1)/(d + 1)
#
# For a qubit:
#
# F_avg = (2 F_process + 1)/3
# ============================================================

d = 2

F_avg = (
    2 * F_process + 1
) / 3.0


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("FINAL RESULTS")
print("========================================")

print(
    f"Process fidelity      = "
    f"{F_process:.10f}"
)

print(
    f"Process fidelity (%)  = "
    f"{100 * F_process:.6f} %"
)

print()

print(
    f"Average gate fidelity = "
    f"{F_avg:.10f}"
)

print(
    f"Average gate fidelity (%) = "
    f"{100 * F_avg:.6f} %"
)

print()

print(
    f"Process infidelity    = "
    f"{1 - F_process:.10e}"
)

print(
    f"Gate infidelity       = "
    f"{1 - F_avg:.10e}"
)


# ============================================================
# TRACE PRESERVATION CHECK
# ============================================================

print("\n========================================")
print("TRACE PRESERVATION CHECK")
print("========================================")

for i, rho in enumerate(input_states):

    rho_out = channel(rho)

    print(
        f"State {i}: "
        f"Tr(input) = {np.trace(rho):.10f}, "
        f"Tr(channel output) = {np.trace(rho_out):.10f}"
    )


# ============================================================
# BLOCH VECTORS
# ============================================================

def bloch_vector(rho):

    x = np.real(
        np.trace(
            rho @ sigmax().full()
        )
    )

    y = np.real(
        np.trace(
            rho @ sigmay().full()
        )
    )

    z = np.real(
        np.trace(
            rho @ sigmaz().full()
        )
    )

    return np.array([x, y, z])


print("\n========================================")
print("INPUT BLOCH VECTORS")
print("========================================")

for i, rho in enumerate(input_states):

    print(
        f"Input {i}: "
        f"{bloch_vector(rho)}"
    )


print("\n========================================")
print("OUTPUT BLOCH VECTORS")
print("========================================")

for i, rho in enumerate(output_states):

    print(
        f"Output {i}: "
        f"{bloch_vector(rho)}"
    )