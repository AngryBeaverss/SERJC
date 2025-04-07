import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# Constants
hbar = 1.0
GHz_to_MHz = 1000
total_time = 50
time_points = np.linspace(0, total_time, 1000)
n_max = 10

# Experimental parameters
omega_qubit_real = 5.0
omega_cavity_real = 5.0
drive_strength_real = 50 / GHz_to_MHz
gamma_spont_real = 1 / GHz_to_MHz
kappa_real = 0.1 / GHz_to_MHz

# Feedback strengths
feedback_strengths = [0.0, 1.0, 2.0]
coupling_strength = 3.25 / GHz_to_MHz
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

# Initial Bell State
rho_bell = np.array([[0, 0, 0, 0],
                     [0, 0.5, 0.5, 0],
                     [0, 0.5, 0.5, 0],
                     [0, 0, 0, 0]], dtype=complex)

rho_cavity_init = np.zeros((n_max, n_max), dtype=complex)
rho_cavity_init[0, 0] = 1.0  # Perfect vacuum state

rho_init = np.kron(rho_bell, rho_cavity_init)
rho_init /= np.trace(rho_init)
rho_init_flat = rho_init.flatten()

# Function to enforce positivity (avoid unphysical states)
def enforce_positivity(rho):
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals[eigvals < 0] = 0  # Clamp negative eigenvalues
    rho_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return rho_fixed / np.trace(rho_fixed)  # Normalize

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

def compute_quantities(rho_m):
    rho2 = rho_m @ rho_m
    purity = np.real(np.trace(rho2))
    w = np.linalg.eigvalsh(rho_m)
    w_pos = w[w > 1e-12]
    entropy = -np.sum(w_pos * np.log2(w_pos)) if len(w_pos) > 0 else 0.0
    coherence = np.sum(np.abs(rho_m)) - np.trace(np.abs(rho_m))
    trace = np.real(np.trace(rho_m))
    return purity, entropy, coherence, trace


# Additional Metrics
def purity_measure(rho):
    return np.trace(rho @ rho).real


def von_neumann_entropy(rho):
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 0]
    return -np.sum(eigvals * np.log2(eigvals))


def fidelity(rho, rho_initial):
    sqrt_rho = scipy.linalg.sqrtm(rho)
    return np.trace(scipy.linalg.sqrtm(sqrt_rho @ rho_initial @ sqrt_rho)).real ** 2


def quantum_fisher_information(rho):
    eigvals, eigvecs = np.linalg.eigh(rho)
    qfi = 0
    for i in range(len(eigvals)):
        for j in range(len(eigvals)):
            if eigvals[i] + eigvals[j] > 0:
                qfi += (2 * (eigvals[i] - eigvals[j]) ** 2) / (eigvals[i] + eigvals[j])
    return qfi


# Store results
results = {beta: {"concurrence": [], "purity": [], "entropy": [], "fidelity": [], "qfi": []} for beta in
           feedback_strengths}

for beta_max in feedback_strengths:
    H_interaction_sweep = coupling_strength * (
            np.kron(np.kron(sigma_plus, sigma_minus), a) +
            np.kron(np.kron(sigma_minus, sigma_plus), a_dagger)
    )
    H_total_sweep = H_qubit_real + H_cavity_real + H_interaction_sweep + H_drive_real


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
            H_feedback = beta * F_rho * np.kron(np.kron(sigma_x, sigma_x), np.eye(n_max))
            drho_dt += -1j / hbar * (H_feedback @ rho - rho @ H_feedback)

        return drho_dt.flatten()


    sol = solve_ivp(lindblad_rhs_sweep, [0, total_time], rho_init_flat, t_eval=time_points, method='RK45')

    interp_solution = interp1d(sol.t, sol.y, kind='linear', axis=1, fill_value="extrapolate")

    for i, t in enumerate(time_points):
        rho_t = interp_solution(t).reshape(4 * n_max, 4 * n_max)
        rho_t = enforce_positivity(rho_t)

        results[beta_max]["concurrence"].append(concurrence_measure(rho_t))
        results[beta_max]["purity"].append(compute_quantities(rho_t)[0])
        results[beta_max]["entropy"].append(compute_quantities(rho_t)[1])
        results[beta_max]["fidelity"].append(fidelity(rho_t, rho_init))
        results[beta_max]["qfi"].append(quantum_fisher_information(rho_t))

# Plot Results

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for beta in feedback_strengths:
    axes[0, 0].plot(time_points, results[beta]["concurrence"], label=f'β = {beta:.2f}')
    axes[0, 1].plot(time_points, results[beta]["purity"], label=f'β = {beta:.2f}')
    axes[1, 0].plot(time_points, results[beta]["entropy"], label=f'β = {beta:.2f}')
    axes[1, 1].plot(time_points, results[beta]["fidelity"], label=f'β = {beta:.2f}')

axes[0, 0].set_title("Concurrence Over Time")
axes[0, 1].set_title("Purity Over Time")
axes[1, 0].set_title("Von Neumann Entropy")
axes[1, 1].set_title("Fidelity")

for ax in axes.flat:
    ax.set_xlabel("Time")
    ax.legend()
    ax.grid()

plt.tight_layout()
plt.show()
