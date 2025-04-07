import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Scaling factors (1 simulation unit = 1 GHz)
GHz_to_MHz = 1000

# Constants
hbar = 1.0
total_time = 50  # ns
time_points = np.linspace(0, total_time, 5000)
n_max = 10  # Cavity Fock states (adjustable)

# Real-world parameters (GHz/MHz)
omega_qubit1 = 5.0            # 5 GHz, qubit 1 frequency
omega_qubit2 = 5.1            # 5.1 GHz, qubit 2 frequency (slight detuning)
omega_cavity = 5.0            # 5 GHz cavity frequency
g1 = 100 / GHz_to_MHz         # 100 MHz coupling strength, qubit 1
g2 = 100 / GHz_to_MHz         # 100 MHz coupling strength, qubit 2
drive_strength = 50 / GHz_to_MHz  # 50 MHz external drive (applied to both)
gamma_spont = 0.7 / GHz_to_MHz  # Reduced to 0.7 MHz to prevent over-damping
kappa = 0.1 / GHz_to_MHz      # 0.1 MHz cavity decay

# Single-qubit operators
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
sigma_minus = np.array([[0, 0], [1, 0]], dtype=complex)
sigma_plus = sigma_minus.T.conj()
I2 = np.eye(2, dtype=complex)

# Two-qubit operators (4x4)
I4 = np.eye(4, dtype=complex)
sigma_x1 = np.kron(sigma_x, I2)
sigma_x2 = np.kron(I2, sigma_x)
sigma_z1 = np.kron(sigma_z, I2)
sigma_z2 = np.kron(I2, sigma_z)
sigma_minus1 = np.kron(sigma_minus, I2)
sigma_minus2 = np.kron(I2, sigma_minus)
sigma_plus1 = sigma_minus1.T.conj()
sigma_plus2 = sigma_minus2.T.conj()

# Cavity operators
a = np.diag(np.sqrt(np.arange(1, n_max)), 1)
a_dagger = a.T

# Full system operators (4 x n_max dimension)
H_qubit1 = 0.5 * omega_qubit1 * np.kron(sigma_z1, np.eye(n_max))
H_qubit2 = 0.5 * omega_qubit2 * np.kron(sigma_z2, np.eye(n_max))
H_cavity = omega_cavity * np.kron(I4, a_dagger @ a)
H_int1 = g1 * (np.kron(sigma_plus1, a) + np.kron(sigma_minus1, a_dagger))
H_int2 = g2 * (np.kron(sigma_plus2, a) + np.kron(sigma_minus2, a_dagger))
H_drive = drive_strength * np.kron(sigma_x1 + sigma_x2, np.eye(n_max))
H_total = H_qubit1 + H_qubit2 + H_cavity + H_int1 + H_int2 + H_drive

# Lindblad operators
L_qubit1 = np.sqrt(gamma_spont) * np.kron(sigma_minus1, np.eye(n_max))
L_qubit2 = np.sqrt(gamma_spont) * np.kron(sigma_minus2, np.eye(n_max))
L_cavity = np.sqrt(kappa) * np.kron(I4, a)
L_list = [L_qubit1, L_qubit2, L_cavity]

# Identity for full system
I_full = np.eye(4 * n_max, dtype=complex)

# Initial state (qubits + cavity)
rho_qubit_init = np.array([[0.4, 0.2, 0.1, 0],
                           [0.2, 0.3, 0.1, 0],
                           [0.1, 0.1, 0.2, 0],
                           [0, 0, 0, 0.1]], dtype=complex)
rho_cavity_init = np.zeros((n_max, n_max), dtype=complex)
rho_cavity_init[0, 0] = 1.0  # Cavity in vacuum
rho_init = np.kron(rho_qubit_init, rho_cavity_init)
rho_init /= np.trace(rho_init)
rho_init_flat = rho_init.flatten()

# Coherence Measure (Corrected)
def coherence_measure(rho):
    rho_qubit = sum(rho[n*4:(n+1)*4, n*4:(n+1)*4] for n in range(n_max))
    coherence_value = np.sum(np.abs(rho_qubit - np.diag(np.diag(rho_qubit)))) / np.sum(np.abs(rho_qubit))
    return coherence_value

# Positivity projection
def project_to_positive(rho):
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    eigenvalues = np.maximum(eigenvalues, 0)
    rho_positive = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.conj().T
    rho_positive /= np.trace(rho_positive)  # Ensure trace normalization
    return rho_positive

# Lindblad RHS with SER feedback and positivity
def lindblad_rhs(t, rho_flat):
    rho = rho_flat.reshape(4 * n_max, 4 * n_max)
    drho_dt = -1j / hbar * (H_total @ rho - rho @ H_total)

    for L in L_list:
        rate = gamma_spont if np.array_equal(L, L_qubit1) or np.array_equal(L, L_qubit2) else kappa
        lindblad_term = L @ rho @ L.conj().T - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L)
        drho_dt += rate * lindblad_term

    # SER Feedback (Refined)
    coherence = coherence_measure(rho)
    beta = min(1.5 * (1 - coherence), 1.5)
    feedback = beta * (I_full - rho) @ L_qubit1 @ rho @ L_qubit1.conj().T @ (I_full - rho)
    feedback += beta * (I_full - rho) @ L_qubit2 @ rho @ L_qubit2.conj().T @ (I_full - rho)
    drho_dt += feedback

    # Ensure trace normalization and positivity
    dt = time_points[1] - time_points[0]
    rho_new = rho + drho_dt * dt
    rho_new /= np.trace(rho_new)
    rho_new = project_to_positive(rho_new)

    return (rho_new.flatten() - rho_flat) / dt

# Solve using RK45
solution = solve_ivp(lindblad_rhs, [0, total_time], rho_init_flat, t_eval=time_points, method='RK45', rtol=1e-6, atol=1e-8)

# Compute purity and coherence
purity = np.real([np.trace(rho.reshape(4 * n_max, 4 * n_max) @ rho.reshape(4 * n_max, 4 * n_max)) for rho in solution.y.T])
coherence = np.array([coherence_measure(rho.reshape(4 * n_max, 4 * n_max)) for rho in solution.y.T])

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(solution.t, purity, label='Purity (2 Qubits)')
plt.plot(solution.t, coherence, label='Coherence (2 Qubits)')
plt.xlabel('Time (ns)')
plt.ylabel('Value')
plt.title('SER Model: Two Qubits Coupled to a Cavity')
plt.legend()
plt.grid(True)
plt.show()
