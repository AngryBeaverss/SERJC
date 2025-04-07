import numpy as np
import matplotlib.pyplot as plt

##############################################################################
#  Circuit QED SER Simulation
#  - Single transmon qubit + single resonator mode
#  - Includes amplitude damping (spontaneous emission, resonator decay)
#  - Includes qubit pure dephasing
#  - Applies SER feedback based on qubit coherence
#
#  NOTE: Frequencies & rates are dimensionless.
#  Typically, you'd scale them so that omega_qubit=1 => ~5 GHz, etc.
##############################################################################

hbar = 1.0
dt = 0.0005           # smaller dt for better numerical stability under stronger feedback
total_time = 50.0
steps = int(total_time / dt)
time = np.linspace(0, total_time, steps)

#---------------------------------------------------------------------------
# 1) SYSTEM PARAMETERS (DIMENSIONLESS)
#    Relative magnitudes set to typical circuit QED ratios:
#    (qubit ~ 1, resonator slightly detuned, couplings ~ 0.05-0.1, decoherence ~ 0.001-0.01)
#---------------------------------------------------------------------------

omega_qubit = 1.0
omega_res   = 1.0   # no detuning
g           = 0.10  # stronger coupling
drive_strength= 1.00   # External drive amplitude on the qubit
gamma_spont   = 0.005  # Qubit spontaneous emission rate (T1-type relaxation)
gamma_phi     = 0.003  # Qubit pure dephasing rate  (T2*-type dephasing)
kappa         = 0.002  # Resonator decay rate
n_max         = 10     # Truncated Fock space dimension for resonator

#---------------------------------------------------------------------------
# 2) OPERATOR DEFINITIONS
#---------------------------------------------------------------------------

# Single-qubit Pauli operators
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
sigma_plus  = sigma_minus.T

# Cavity annihilation operator a in truncated Fock basis
a = np.diag(np.sqrt(np.arange(1, n_max)), 1)
a_dagger = a.conj().T

# Identity operators
Id_2    = np.eye(2, dtype=complex)
Id_nmax = np.eye(n_max, dtype=complex)
Id_full = np.eye(2 * n_max, dtype=complex)

#---------------------------------------------------------------------------
# 3) COMPOSITE HAMILTONIAN
#---------------------------------------------------------------------------

# H_qubit = (1/2)*omega_qubit * sigma_z
H_qubit = 0.5 * omega_qubit * np.kron(sigma_z, Id_nmax)

# H_res   = omega_res * a_dagger*a
H_res = omega_res * np.kron(Id_2, a_dagger @ a)

# Jaynes–Cummings coupling: g ( sigma_plus*a + sigma_minus*a_dagger )
H_int = g * (np.kron(sigma_plus, a) + np.kron(sigma_minus, a_dagger))

# Drive on the qubit (x direction)
H_drive = drive_strength * np.kron(sigma_x, Id_nmax)

# Total Hamiltonian
H = H_qubit + H_res + H_int + H_drive

#---------------------------------------------------------------------------
# 4) LINDBLAD OPERATORS
#---------------------------------------------------------------------------

# Qubit amplitude damping (T1)
L_qubit_decay = np.sqrt(gamma_spont) * np.kron(sigma_minus, Id_nmax)

# Qubit pure dephasing: L_phi = sqrt(gamma_phi)*sigma_z
# Typically the Lindblad form is L_phi*rho*L_phi - ...
L_qubit_deph  = np.sqrt(gamma_phi) * np.kron(sigma_z, Id_nmax)

# Resonator photon decay
L_res_decay   = np.sqrt(kappa) * np.kron(Id_2, a)

# Collect them in a list
L_list = [L_qubit_decay, L_qubit_deph, L_res_decay]

#---------------------------------------------------------------------------
# 5) INITIAL STATE
#    Qubit partially coherent, resonator in vacuum
#    Optionally add small offset for other Fock states
#---------------------------------------------------------------------------

# Qubit: 2x2
rho_qubit_init = np.array([
    [0.6, 0.4],
    [0.4, 0.4]
], dtype=complex)

# Resonator: n_max x n_max
rho_res_init = np.zeros((n_max, n_max), dtype=complex)
rho_res_init[0, 0] = 1.0  # vacuum

# Kronecker product => full system
rho_init = np.kron(rho_qubit_init, rho_res_init)

# Optionally add a small uniform background in the resonator
for i in range(1, n_max):
    rho_init += 0.002 * np.kron(Id_2, np.outer(np.ones(n_max)/n_max, np.ones(n_max)/n_max))

rho_init /= np.trace(rho_init)  # normalize

rho = rho_init.copy()

#---------------------------------------------------------------------------
# 6) HELPER FUNCTIONS
#---------------------------------------------------------------------------

def enforce_positivity(rho_matrix):
    """Project onto positive semidefinite space and normalize trace."""
    w, v = np.linalg.eigh(rho_matrix)
    w_clamped = np.clip(w, 0, None)
    rho_psd = v @ np.diag(w_clamped) @ v.conj().T
    return rho_psd / np.trace(rho_psd)

def partial_trace_qubit(rho_matrix, n_fock):
    """Partial trace over resonator to get 2x2 qubit state."""
    dim_q = 2
    rho_q = np.zeros((dim_q, dim_q), dtype=complex)
    for n in range(n_fock):
        block = rho_matrix[n*dim_q:(n+1)*dim_q, n*dim_q:(n+1)*dim_q]
        rho_q += block
    return rho_q

def compute_coherence_qubit(rho_matrix, n_fock):
    """Compute qubit coherence from partial trace of full state."""
    rho_q = partial_trace_qubit(rho_matrix, n_fock)
    return np.abs(rho_q[0,1]) + np.abs(rho_q[1,0])

def compute_purity(rho_matrix):
    """Return purity = Tr(rho^2)."""
    return np.real(np.trace(rho_matrix @ rho_matrix))

#---------------------------------------------------------------------------
# 7) EVOLUTION FUNCTION (Lindblad + SER feedback in a single step)
#---------------------------------------------------------------------------

def step_evolution(rho, H, L_ops, dt, n_fock):
    # 1) Hamiltonian commutator
    commutator = -1j/hbar * (H @ rho - rho @ H)
    drho_dt = commutator

    # 2) Lindblad terms
    for L_op in L_ops:
        # Distinguish which rate to apply
        # (Optional – you could embed rates in L_op itself.)
        if np.allclose(L_op, L_qubit_decay):
            rate = gamma_spont
        elif np.allclose(L_op, L_qubit_deph):
            rate = gamma_phi
        else:
            rate = kappa
        LR = L_op @ rho @ L_op.conj().T
        drho_dt += rate * (LR - 0.5*(L_op.conj().T@L_op@rho + rho@L_op.conj().T@L_op))

    # 3) SER feedback
    coherence_q = compute_coherence_qubit(rho, n_fock)
    # example function F(rho)
    # in step_evolution
    F_rho = 1.0  # remove the exponential damp
    beta = min(2.0 * (1 - coherence_q), 5.0)
    # We apply feedback with the qubit decay operator to re-inject amplitude
    # (Could also pick a different or expanded L_op for SER)
    feedback_op = L_qubit_decay

    # (I - rho) L rho L^\dagger (I - rho)
    temp = (Id_full - rho) @ feedback_op @ rho @ feedback_op.conj().T @ (Id_full - rho)
    drho_dt += beta * F_rho * temp

    # 4) Update and enforce positivity
    rho_next = rho + drho_dt*dt
    rho_psd  = enforce_positivity(rho_next)
    return rho_psd

#---------------------------------------------------------------------------
# 8) MAIN SIMULATION LOOP
#---------------------------------------------------------------------------

purities = np.zeros(steps)
coherences = np.zeros(steps)

for i in range(steps):
    # Evolve
    rho = step_evolution(rho, H, L_list, dt, n_max)

    # Record
    purities[i] = compute_purity(rho)
    coherences[i] = compute_coherence_qubit(rho, n_max)

#---------------------------------------------------------------------------
# 9) PLOTTING
#---------------------------------------------------------------------------

plt.figure(figsize=(10,5))
plt.plot(time, purities, label="Purity")
plt.plot(time, coherences, label="Qubit Coherence")
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Circuit QED SER Simulation (Single Qubit + Single Mode)")
plt.legend()
plt.grid(True)
plt.show()
