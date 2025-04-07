import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
hbar = 1.0
GHz_to_MHz = 1000
total_time = 100  # Extend if needed
time_points = np.linspace(0, total_time, 100)  # More time resolution
n_max = 10  # Truncated cavity levels

# Experimental parameters
omega_qubit_real = 5.0
omega_cavity_real = 5.0
drive_strength_real = 10 / GHz_to_MHz
gamma_spont_real = 1 / GHz_to_MHz
kappa_real = 0.1 / GHz_to_MHz

# Test different feedback strengths
feedback_strengths = [0.25, 0.5, 0.75, 1.0]

# Use a mid-range coupling value to test feedback effectiveness
coupling_strength = 3.25 / GHz_to_MHz

# Feedback delay for experimental feasibility
tau_f = 1.0

# Define Pauli Y matrix for the spin-flip operation
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
# Compute sigma_y ⊗ sigma_y for two qubits
sigma_y_tensor = np.kron(sigma_y, sigma_y)

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

# Work extraction operator
W_op = np.kron(np.kron(sigma_plus, sigma_minus), a_dagger) + np.kron(np.kron(sigma_minus, sigma_plus), a)

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
rho_init = np.kron(rho_bell, rho_cavity_init)  # Uncorrelated Bell state ⊗ Vacuum
rho_init = (rho_init + rho_init.conj().T) / 2  # Enforce Hermitian symmetry
rho_init /= np.trace(rho_init)  # Properly normalize
rho_init_flat = rho_init.flatten()

# Function to measure entanglement (Concurrence)
def concurrence_measure(rho, time_step=None):
    # Step 1: Trace out the cavity to get the reduced density matrix for the qubits
    rho_qubits = np.trace(rho.reshape(4, n_max, 4, n_max), axis1=1, axis2=3)

    # Ensure rho_qubits is Hermitian and normalized (due to numerical errors)
    rho_qubits = (rho_qubits + rho_qubits.conj().T) / 2  # Enforce Hermitian
    rho_qubits = rho_qubits / np.trace(rho_qubits)  # Normalize trace to 1

    # Step 2: Compute the spin-flipped density matrix
    # rho^* is the complex conjugate of rho_qubits
    rho_star = np.conj(rho_qubits)
    # Compute (sigma_y ⊗ sigma_y) rho^* (sigma_y ⊗ sigma_y)
    rho_tilde = sigma_y_tensor @ rho_star @ sigma_y_tensor

    # Step 3: Compute sqrt(rho)
    # Use eigenvalue decomposition for numerical stability
    eigvals, eigvecs = np.linalg.eigh(rho_qubits)
    # Clamp small negative eigenvalues due to numerical errors
    eigvals = np.maximum(eigvals, 0)
    sqrt_rho = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.conj().T

    # Step 4: Compute R = sqrt(sqrt(rho) rho_tilde sqrt(rho))
    temp = sqrt_rho @ rho_tilde @ sqrt_rho
    # Compute sqrt(temp) via eigenvalue decomposition
    eigvals_temp, eigvecs_temp = np.linalg.eigh(temp)
    eigvals_temp = np.maximum(eigvals_temp, 0)  # Avoid negative eigenvalues
    sqrt_temp = eigvecs_temp @ np.diag(np.sqrt(eigvals_temp)) @ eigvecs_temp.conj().T

    # Step 5: Compute the eigenvalues of R (which is sqrt_temp)
    # Since we already have the eigenvalues from the decomposition, use them
    lambdas = np.sqrt(np.maximum(eigvals_temp, 0))  # Eigenvalues of R

    # Step 6: Sort eigenvalues in descending order and compute concurrence
    lambdas = np.sort(lambdas)[::-1]  # Sort in descending order
    concurrence = max(0, lambdas[0] - lambdas[1] - lambdas[2] - lambdas[3])
    if concurrence > 1:
        print(f"Warning: Concurrence exceeded 1. Value: {concurrence}")

    return concurrence


def von_neumann_entropy(rho):
    """Compute von Neumann entropy S(ρ) = -Tr(ρ log ρ), ensuring physical validity."""
    eigvals = np.linalg.eigvalsh(rho)

    # Small numerical errors can lead to slightly negative eigenvalues.
    eigvals = np.maximum(eigvals, 1e-10)  # Ensure non-negative eigenvalues

    entropy = -np.sum(eigvals * np.log(eigvals))

    # **Diagnostic Check**: Print warning if entropy is negative
    if entropy < 0:
        print(f"Warning: Negative entropy detected. Clamping to zero. Raw value: {entropy}")

    return max(0, entropy)  # Ensure entropy is physically valid

def total_energy(rho):
    """Compute total system energy E(ρ) = Tr(H_total ρ)"""
    return np.real(np.trace(H_total_sweep @ rho))

# Function to enforce positivity (avoid unphysical states)
def enforce_positivity(rho):
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals[eigvals < 0] = 0  # Clamp negative eigenvalues
    rho_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return rho_fixed / np.trace(rho_fixed)  # Normalize

# **Function to Compute Extractable Work**
def extractable_work(rho):
    return np.real(np.trace(W_op @ rho)) - von_neumann_entropy(rho)


# Store results for plotting
concurrence_over_time = {beta: [] for beta in feedback_strengths}
work_over_time = {beta: [] for beta in feedback_strengths}
entropy_over_time = {beta: [] for beta in feedback_strengths}
energy_over_time = {beta: [] for beta in feedback_strengths}

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
            entanglement = max(0, min(1, concurrence_measure(rho)))
            F_rho = np.exp(-2 * (1 - entanglement))
            beta = min(beta_max * (1 - entanglement) * np.exp(-entanglement), 0.10)
            H_feedback = beta * F_rho * (
                    0.01 * np.kron(np.kron(sigma_x, sigma_x), np.eye(n_max)) +
                    100 * np.kron(np.eye(4), a + a_dagger)
            )
            drho_dt += -1j / hbar * (H_feedback @ rho - rho @ H_feedback)

        return drho_dt.flatten()


    # Solve ODE
    sol = solve_ivp(lindblad_rhs_sweep, [0, total_time], rho_init_flat,
                    t_eval=time_points, method='RK45')

    # Compute concurrence and work at each time step
    for i in range(len(time_points)):
        rho_t = sol.y[:, i].reshape(4 * n_max, 4 * n_max)
        rho_t = enforce_positivity(rho_t)
        concurrence_over_time[beta_max].append(concurrence_measure(rho_t, time_points[i]))
        work_over_time[beta_max].append(extractable_work(rho_t))
        rho_qubits = np.trace(rho_t.reshape(4, n_max, 4, n_max), axis1=1, axis2=3)
        rho_qubits = (rho_qubits + rho_qubits.conj().T) / 2  # Ensure Hermitian
        rho_qubits = rho_qubits / np.trace(rho_qubits)  # Normalize trace to 1
        entropy_over_time[beta_max].append(von_neumann_entropy(rho_qubits))
        energy_over_time[beta_max].append(total_energy(rho_t))


# Entanglement evolution across β values
plt.figure(figsize=(10, 6))
for beta in feedback_strengths:
    plt.plot(time_points, concurrence_over_time[beta], label=f'β = {beta:.2f}')
plt.xlabel("Time")
plt.ylabel("Concurrence")
plt.title("Time Evolution of Entanglement Under SER Feedback (β Sweep)")
plt.legend()
plt.grid(True)
plt.show()

# Extractable Work across β values
plt.figure(figsize=(10, 6))
for beta in feedback_strengths:
    plt.plot(time_points, work_over_time[beta], label=f'β = {beta:.2f}')
plt.xlabel("Time")
plt.ylabel("Extractable Work")
plt.title("Extractable Work Under SER Feedback (β Sweep)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
for beta in feedback_strengths:
    plt.plot(time_points, entropy_over_time[beta], label=f'β = {beta:.2f}')
plt.xlabel("Time")
plt.ylabel("Entropy")
plt.title("Entropy Evolution Under Feedback")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
for beta in feedback_strengths:
    plt.plot(time_points, energy_over_time[beta], label=f'β = {beta:.2f}')
plt.xlabel("Time")
plt.ylabel("Total Energy")
plt.title("Total System Energy Over Time")
plt.legend()
plt.grid(True)
plt.show()

# Test the concurrence function
def test_concurrence():
    # Bell state: (|00⟩ + |11⟩)/sqrt(2)
    bell_state = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)])
    rho_bell = np.outer(bell_state, bell_state.conj())
    # Simulate rho with a trivial cavity state for testing
    rho_test = np.kron(rho_bell, np.eye(n_max)/n_max)
    print("Concurrence of Bell state:", concurrence_measure(rho_test))

    # Separable state: |00⟩⟨00|
    separable_state = np.array([1, 0, 0, 0])
    rho_separable = np.outer(separable_state, separable_state.conj())
    rho_test = np.kron(rho_separable, np.eye(n_max)/n_max)
    print("Concurrence of separable state:", concurrence_measure(rho_test))

# Run the test
test_concurrence()