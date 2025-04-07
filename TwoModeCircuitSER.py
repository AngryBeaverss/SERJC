import numpy as np
import matplotlib.pyplot as plt

# ======== 1) BASIC PARAMETERS ========
hbar = 1.0
dt = 0.001
total_time = 25.0
steps = int(total_time / dt)
time = np.linspace(0, total_time, steps)

omega_q = 1.0
omega_res = 1.0
g = 0.05
gamma_spont = 0.003
gamma_phi = 0.001
kappa = 0.001
drive_strength = 0.0

n_max = 5
dim_q = 2
dim_c = n_max
dim_full = dim_q * dim_c

# ======== 2) OPERATORS ========
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
sigma_plus = sigma_minus.conj().T


def fock_annihilation(n):
    return np.diag(np.sqrt(np.arange(1, n)), 1)


a = fock_annihilation(n_max)
a_dag = a.conj().T


def kron2(A, B):
    return np.kron(A, B)


Id_q = np.eye(2, dtype=complex)
Id_c = np.eye(n_max, dtype=complex)
Id_full = np.eye(dim_full, dtype=complex)

# Hamiltonian pieces
H_q = 0.5 * omega_q * kron2(sigma_z, Id_c)
H_res = omega_res * kron2(Id_q, a_dag @ a)
H_int = g * (kron2(sigma_plus, a) + kron2(sigma_minus, a_dag))
H_drive = drive_strength * kron2(sigma_x, Id_c)
H = H_q + H_res + H_int + H_drive

# Lindblad ops
L_q_decay = np.sqrt(gamma_spont) * kron2(sigma_minus, Id_c)
L_q_deph = np.sqrt(gamma_phi) * kron2(sigma_z, Id_c)
L_c_decay = np.sqrt(kappa) * kron2(Id_q, a)

L_list = [L_q_decay, L_q_deph, L_c_decay]

# ======== 3) INITIAL STATE ========
rho_q_init = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex)  # partial superposition
rho_c_init = np.zeros((n_max, n_max), dtype=complex)
rho_c_init[0, 0] = 1.0  # vacuum
rho_init = np.kron(rho_q_init, rho_c_init)
rho_init /= np.trace(rho_init)

rho = rho_init.copy()


# ======== 4) HELPER FUNCTIONS ========
def enforce_positivity(r):
    w, v = np.linalg.eigh(r)
    w_clamp = np.clip(w, 0, None)
    r_psd = v @ np.diag(w_clamp) @ v.conj().T
    return r_psd / np.trace(r_psd)


def partial_trace_qubit(r):
    """Trace out the cavity (n_max) to get 2x2 qubit state."""
    rho_q = np.zeros((2, 2), dtype=complex)
    for n in range(n_max):
        block = r[n * 2:(n + 1) * 2, n * 2:(n + 1) * 2]
        rho_q += block
    return rho_q


def qubit_coherence(r):
    rho_q = partial_trace_qubit(r)
    return np.abs(rho_q[0, 1]) + np.abs(rho_q[1, 0])


def purity(r):
    return np.real(np.trace(r @ r))


# ======== 5) MAIN EVOLUTION WITH SER FEEDBACK (sigma_plus) ========

# 1) Construct the feedback operator as sigma_+ instead of sigma_-
L_plus = kron2(sigma_plus, Id_c)  # raising operator
feedback_max = 2.0

coherence_vals = np.zeros(steps)
purity_vals = np.zeros(steps)

for i in range(steps):
    # (a) standard Lindblad + Hamiltonian step
    comm = -1j / hbar * (H @ rho - rho @ H)
    drho_dt = comm

    # add Lindblad
    for L_op in L_list:
        LrhoL = L_op @ rho @ L_op.conj().T
        drho_dt += LrhoL - 0.5 * (L_op.conj().T @ L_op @ rho + rho @ L_op.conj().T @ L_op)

    rho_temp = rho + drho_dt * dt
    rho_temp = enforce_positivity(rho_temp)

    # (b) measure qubit coherence
    c = qubit_coherence(rho_temp)
    # e.g. simple F_rho
    F_rho = 1.0  # or np.exp(-2*(1-c)) if you want
    # define a bigger feedback if c < 0.5, for example
    beta = min(1.0 * (1.0 - c), feedback_max)

    # (c) SER feedback using sigma_plus
    # direct off-diagonal pumping
    temp = (Id_full - rho_temp) @ L_plus @ rho_temp @ L_plus.conj().T @ (Id_full - rho_temp)
    rho_new = rho_temp + beta * F_rho * temp * dt
    rho_new = enforce_positivity(rho_new)

    rho = rho_new

    # (d) record
    coherence_vals[i] = qubit_coherence(rho)
    purity_vals[i] = purity(rho)

# ======== 6) PLOT ========
plt.figure(figsize=(8, 5))
plt.plot(time, purity_vals, label='Purity')
plt.plot(time, coherence_vals, label='Qubit Coherence')
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)
plt.legend()
plt.title("Direct Off-Diagonal Pumping with sigma+ in SER Feedback")
plt.show()
