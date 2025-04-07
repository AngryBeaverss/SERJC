
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

##############################################################################
# Configuration Parameters
##############################################################################
feedback_mode = 'SER'  # Choose: 'SER' or 'TQG'
lambda_coeff = 0.02    # Only relevant for TQG mode

hbar = 1.0
GHz_to_MHz = 1000
total_time = 200.0
time_points = np.linspace(0, total_time, 400)
n_max = 15

omega_qubit_real = 5.0
omega_cavity_real = 5.0
drive_strength_real = 10 / GHz_to_MHz
gamma_spont_real = 1 / GHz_to_MHz
kappa_real = 0.1 / GHz_to_MHz
feedback_strengths = [0.0, 0.5, 1.0]
coupling_strengths = [20 / GHz_to_MHz]
tau_f = 1.0

sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
sigma_plus = sigma_minus.T
a = np.diag(np.sqrt(np.arange(1, n_max)), 1)
a_dagger = a.T
dim_system = 4 * n_max

##############################################################################
# Helper Functions
##############################################################################
def enforce_positivity(rho):
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.clip(eigvals, 1e-10, None)
    rho_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
    return rho_fixed / np.trace(rho_fixed)

def adaptive_feedback(beta_max, entanglement):
    return min(beta_max * (1 - entanglement) * np.exp(-entanglement), 0.02)

def partial_trace_cavity(rho_full, n_qubits=4, n_cavity=n_max):
    rho_4d = rho_full.reshape(n_qubits, n_cavity, n_qubits, n_cavity)
    return np.trace(rho_4d, axis1=1, axis2=3)

def compute_concurrence(rho_2q):
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_yy = np.kron(sigma_y, sigma_y)
    rho_star = np.conjugate(rho_2q)
    rho_tilde = sigma_yy @ rho_star @ sigma_yy
    product = rho_2q @ rho_tilde
    eigenvals = np.sort(np.real(np.linalg.eigvals(product)))[::-1]
    sqrt_vals = np.sqrt(np.clip(eigenvals, 0, None))
    return max(0.0, sqrt_vals[0] - sqrt_vals[1] - sqrt_vals[2] - sqrt_vals[3])

def compute_ergotropy(rho, H):
    energy = np.trace(rho @ H).real
    eigvals_rho, eigvecs_rho = np.linalg.eigh(rho)
    sorted_indices = np.argsort(eigvals_rho)[::-1]
    eigvals_rho = eigvals_rho[sorted_indices]
    eigvecs_rho = eigvecs_rho[:, sorted_indices]
    eigvals_H, eigvecs_H = np.linalg.eigh(H)
    sorted_indices_H = np.argsort(eigvals_H)
    eigvecs_H = eigvecs_H[:, sorted_indices_H]
    rho_passive = sum(eigvals_rho[i] * np.outer(eigvecs_H[:, i], eigvecs_H[:, i].conj()) for i in range(len(eigvals_rho)))
    return energy - np.trace(rho_passive @ H).real

def compute_expectation(rho, operator):
    return np.real(np.trace(rho @ operator))

##############################################################################
# Simulation Routine
##############################################################################
def run_simulation(g, beta_max):
    H_drive = drive_strength_real * np.kron(np.kron(sigma_x, np.eye(2)), np.eye(n_max))
    H_int = g * (np.kron(np.kron(sigma_plus, sigma_minus), a) + np.kron(np.kron(sigma_minus, sigma_plus), a_dagger))
    H_base = H_drive + H_int

    L_qubit = np.sqrt(gamma_spont_real) * np.kron(np.kron(sigma_minus, np.eye(2)), np.eye(n_max))
    L_cavity = np.sqrt(kappa_real) * np.kron(np.eye(4), a)
    L_list = [L_qubit, L_cavity]

    psi_bell = (np.kron([1, 0], [0, 1]) + np.kron([0, 1], [1, 0])) / np.sqrt(2)
    rho_bell = np.outer(psi_bell, psi_bell.conj())
    rho_init = np.zeros((dim_system, dim_system), dtype=complex)
    rho_init[:4, :4] = rho_bell
    rho_init_flat = np.concatenate([rho_init.real.flatten(), rho_init.imag.flatten()])

    def rhs(t, rho_flat):
        half = len(rho_flat) // 2
        rho = rho_flat[:half].reshape(dim_system, dim_system) + 1j * rho_flat[half:].reshape(dim_system, dim_system)
        drho = -1j / hbar * (H_base @ rho - rho @ H_base)
        for L in L_list:
            drho += L @ rho @ L.conj().T - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L)
        if t >= tau_f:
            rho_2q = partial_trace_cavity(rho)
            if feedback_mode == 'SER':
                beta = adaptive_feedback(beta_max, compute_concurrence(rho_2q))
                H_fb = beta * (np.kron(np.kron(sigma_x, sigma_x), np.eye(n_max)) + np.kron(np.eye(4), (a + a_dagger)))
                drho += -1j * (H_fb @ rho - rho @ H_fb)
            elif feedback_mode == 'TQG':
                purity = np.trace(rho @ rho).real
                A_mu = lambda_coeff * (1 - purity)
                H_gf = A_mu * (np.kron(np.kron(sigma_z, sigma_z), np.eye(n_max)) + np.kron(np.eye(4), (a - a_dagger)))
                drho += -1j * (H_gf @ rho - rho @ H_gf)
        return np.concatenate([drho.real.flatten(), drho.imag.flatten()])

    sol = solve_ivp(rhs, [0, total_time], rho_init_flat, t_eval=time_points, atol=1e-8, rtol=1e-6)

    ergotropy, concurrence, photon_num, qubit_pop, purity_list = [], [], [], [], []
    a_dagger_a = np.kron(np.eye(4), a_dagger @ a)
    qubit_op = np.kron(np.kron(sigma_plus @ sigma_minus, np.eye(2)), np.eye(n_max))

    for i in range(len(sol.t)):
        rho_t = sol.y[:dim_system**2, i] + 1j * sol.y[dim_system**2:, i]
        rho_t = enforce_positivity(rho_t.reshape(dim_system, dim_system))
        rho_2q = partial_trace_cavity(rho_t)
        ergotropy.append(compute_ergotropy(rho_t, H_base))
        concurrence.append(compute_concurrence(rho_2q))
        photon_num.append(compute_expectation(rho_t, a_dagger_a))
        qubit_pop.append(compute_expectation(rho_t, qubit_op))
        purity_list.append(np.trace(rho_2q @ rho_2q).real)

    coherence_contrast = max(purity_list) - min(purity_list)
    print(f"🧪 Coherence Contrast (Mode = {feedback_mode}): {coherence_contrast:.6f}")
    return sol.t, ergotropy, concurrence, photon_num, qubit_pop

##############################################################################
# Execution
##############################################################################
for g in coupling_strengths:
    for beta_max in feedback_strengths:
        t, W, C, n_cav, Pq = run_simulation(g, beta_max)

        plt.figure(figsize=(10, 6))
        plt.plot(t, C, label='Concurrence')
        plt.plot(t, W, label='Ergotropy')
        plt.plot(t, n_cav, label='Photon Number')
        plt.plot(t, Pq, label='Qubit Population')
        plt.title(f'Simulation Results (g = {g*GHz_to_MHz:.2f} MHz, β = {beta_max:.2f})')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
