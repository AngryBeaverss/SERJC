import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

# Constants
hbar = 1.0
GHz_to_MHz = 1000
total_time = 50  # Extend if needed
time_points = np.linspace(0, total_time, 500)  # More time resolution
n_max = 10  # Truncated cavity levels

# Experimental parameters
omega_qubit_real = 5.0
omega_cavity_real = 5.0
drive_strength_real = 50 / GHz_to_MHz
gamma_spont_real = 1 / GHz_to_MHz
kappa_real = 0.1 / GHz_to_MHz

# Test different feedback strengths
feedback_strengths = [0.0, 3.0, 6.0, 9.0, 12.0]

# Use a mid-range coupling value to test feedback effectiveness
coupling_strength = 3.25 / GHz_to_MHz

# Feedback delay for experimental feasibility
tau_f = 3.0

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
sigma_plus = sigma_minus.T
a = np.diag(np.sqrt(np.arange(1, n_max)), 1)
a_dagger = a.T

# Two-qubit system interacting via a shared cavity
H_qubit_real = 0.5 * omega_qubit_real * np.kron(np.kron(sigma_z, sigma_z), np.eye(n_max))
H_cavity_real = omega_cavity_real * np.kron(np.eye(4), a_dagger @ a)
H_drive_real = drive_strength_real * np.kron(np.kron(sigma_x, sigma_x), np.eye(n_max))

# Lindblad operators
L_qubit_real = np.sqrt(gamma_spont_real) * np.kron(np.kron(sigma_minus, np.eye(2)), np.eye(n_max))
L_cavity_real = np.sqrt(kappa_real) * np.kron(np.eye(4), a)
L_list_real = [L_qubit_real, L_cavity_real]

# Identity operator
I = np.eye(4 * n_max, dtype=complex)

# Initial Bell State: |ψ⟩ = (|01⟩ + |10⟩) / sqrt(2)
rho_bell = np.array([[0, 0, 0, 0],
                     [0, 0.5, 0.5, 0],
                     [0, 0.5, 0.5, 0],
                     [0, 0, 0, 0]], dtype=complex)

# Cavity mostly vacuum
rho_cavity_init = np.zeros((n_max, n_max), dtype=complex)
rho_cavity_init[0, 0] = 1.0  # Perfect vacuum state

# Full system initial state
rho_init = np.kron(rho_bell, rho_cavity_init)
rho_init /= np.trace(rho_init)
rho_init_flat = rho_init.flatten()

# Function to measure entanglement (Concurrence)
def concurrence_measure(rho):
    rho_qubits = np.trace(rho.reshape(4, n_max, 4, n_max), axis1=1, axis2=3)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_y_sigma_y = np.kron(sigma_y, sigma_y)
    rho_tilde = np.dot(np.dot(sigma_y_sigma_y, np.conj(rho_qubits)), sigma_y_sigma_y)
    R = np.sqrt(np.dot(np.sqrt(rho_qubits), np.dot(rho_tilde, np.sqrt(rho_qubits))))
    lambdas = np.sqrt(np.abs(np.linalg.eigvals(R)))
    lambdas = np.sort(lambdas)[::-1]  # Sort in descending order
    return max(0, lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3])

# Function to enforce positivity (avoid unphysical states)
def enforce_positivity(rho):
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals[eigvals < 0] = 0  # Clamp negative eigenvalues
    rho_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return rho_fixed / np.trace(rho_fixed)  # Normalize

# Store results for plotting
concurrence_over_time = {beta: [] for beta in feedback_strengths}

for beta_max in feedback_strengths:
    H_interaction_sweep = coupling_strength * (
        np.kron(np.kron(sigma_plus, sigma_minus), a) +
        np.kron(np.kron(sigma_minus, sigma_plus), a_dagger)
    )
    H_total_sweep = H_qubit_real + H_cavity_real + H_interaction_sweep + H_drive_real

    # Define RHS for simulation with feedback
    def lindblad_rhs_sweep(t, rho_flat):
        rho = rho_flat.reshape(4 * n_max, 4 * n_max)
        drho_dt = -1j / hbar * (H_total_sweep @ rho - rho @ H_total_sweep)

        for L in L_list_real:
            rate = gamma_spont_real if L is L_qubit_real else kappa_real
            lindblad_term = L @ rho @ L.conj().T - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L)
            drho_dt += rate * lindblad_term

        if t >= tau_f:
            rho = enforce_positivity(rho)
            entanglement = concurrence_measure(rho)
            F_rho = np.exp(-2 * (1 - entanglement))
            beta = min(beta_max * (1 - entanglement), beta_max)

            # Feedback applied as an entanglement-preserving Hamiltonian correction
            H_feedback = beta * F_rho * np.kron(np.kron(sigma_x, sigma_x), np.eye(n_max))
            drho_dt += -1j / hbar * (H_feedback @ rho - rho @ H_feedback)

        return drho_dt.flatten()

    # Solve ODE
    sol = solve_ivp(lindblad_rhs_sweep, [0, total_time], rho_init_flat, t_eval=time_points, method='RK45')

    # Compute concurrence at each time step
    for i in range(len(time_points)):
        rho_t = sol.y[:, i].reshape(4 * n_max, 4 * n_max)
        rho_t = enforce_positivity(rho_t)
        concurrence_over_time[beta_max].append(concurrence_measure(rho_t))

# Plot concurrence over time for different feedback strengths
plt.figure(figsize=(10, 6))
for beta in feedback_strengths:
    plt.plot(time_points, concurrence_over_time[beta], label=f'β = {beta:.2f}')
plt.xlabel("Time")
plt.ylabel("Concurrence")
plt.title("Time Evolution of Entanglement Under SER Feedback")
plt.legend()
plt.grid(True)
plt.show()
