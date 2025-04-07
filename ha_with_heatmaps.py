import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ============================================================================
# 1) Global Constants and Simulation Settings
# ============================================================================
hbar = 1.0
GHz_to_MHz = 1000

# Time and Hilbert space settings
total_time = 200.0
time_points = np.linspace(0, total_time, 400)
n_max = 15  # Fock states for the cavity
dim_qubit = 4  # Two-qubit subspace (Bell state occupies a 4-dim space)
dim_system = dim_qubit * n_max

# Experimental-like parameters
omega_qubit_real = 5.0
omega_cavity_real = 5.0
drive_strength_real = 10 / GHz_to_MHz
gamma_spont_real = 1 / GHz_to_MHz
kappa_real = 0.1 / GHz_to_MHz

# Coupling strengths and feedback levels
coupling_strengths = [20 / GHz_to_MHz, 50 / GHz_to_MHz]  # Weak and strong
feedback_strengths = [0.0, 0.5]  # Feedback levels to test
tau_f_default = 1.0  # Default feedback delay

# Pauli matrices and cavity operators
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
sigma_plus = sigma_minus.T
a = np.diag(np.sqrt(np.arange(1, n_max)), 1)
a_dagger = a.T


# ============================================================================
# 2) Helper Functions
# ============================================================================

def enforce_positivity(rho):
    """Fix small numerical negativity in the density matrix."""
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.clip(eigvals, 1e-10, None)
    rho_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
    return rho_fixed / np.trace(rho_fixed)


def adaptive_feedback(beta_max, entanglement):
    """Adaptive feedback scaling based on concurrence."""
    return min(beta_max * (1 - entanglement) * np.exp(-entanglement), 0.02)


def partial_trace_cavity(rho_full, n_qubits=4, n_cavity=n_max):
    """Trace out the cavity degrees of freedom."""
    rho_reshaped = rho_full.reshape(n_qubits, n_cavity, n_qubits, n_cavity)
    return np.trace(rho_reshaped, axis1=1, axis2=3)


def compute_concurrence(rho_2q):
    """Compute the Wootters concurrence for a 4x4 two-qubit density matrix."""
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_yy = np.kron(sigma_y, sigma_y)
    rho_star = np.conjugate(rho_2q)
    rho_tilde = sigma_yy @ rho_star @ sigma_yy
    product = rho_2q @ rho_tilde
    eigenvals = np.sort(np.linalg.eigvals(product).real)[::-1]
    sqrt_vals = np.sqrt(np.clip(eigenvals, 0, None))
    return max(0.0, sqrt_vals[0] - sqrt_vals[1] - sqrt_vals[2] - sqrt_vals[3])


def compute_ergotropy(rho, H):
    """Compute the extractable work (ergotropy) from the density matrix and Hamiltonian."""
    energy = np.trace(rho @ H).real
    eigvals_rho, eigvecs_rho = np.linalg.eigh(rho)
    sorted_indices = np.argsort(eigvals_rho)[::-1]
    eigvals_rho = eigvals_rho[sorted_indices]
    eigvecs_rho = eigvecs_rho[:, sorted_indices]
    eigvals_H, eigvecs_H = np.linalg.eigh(H)
    sorted_indices_H = np.argsort(eigvals_H)
    eigvals_H = eigvals_H[sorted_indices_H]
    eigvecs_H = eigvecs_H[:, sorted_indices_H]
    rho_passive = sum(eigvals_rho[i] * np.outer(eigvecs_H[:, i], eigvecs_H[:, i].conj())
                      for i in range(len(eigvals_rho)))
    passive_energy = np.trace(rho_passive @ H).real
    return energy - passive_energy


def compute_cavity_occupation(rho, a_dagger_a_full):
    """Compute the expectation value of the cavity photon number operator."""
    return np.real(np.trace(rho @ a_dagger_a_full))


def compute_qubit_population(rho, sigma_plus_sigma_minus_full):
    """Compute the expectation value of the qubit excitation number operator."""
    return np.real(np.trace(rho @ sigma_plus_sigma_minus_full))


# ============================================================================
# 3) System Setup Functions
# ============================================================================

def get_system_operators(g):
    """Build the Hamiltonian and Lindblad operators for a given coupling strength."""
    H_drive_qubit = drive_strength_real * np.kron(np.kron(sigma_x, np.eye(2)), np.eye(n_max))
    L_qubit = np.sqrt(gamma_spont_real) * np.kron(np.kron(sigma_minus, np.eye(2)), np.eye(n_max))
    L_cavity = np.sqrt(kappa_real) * np.kron(np.eye(4), a)
    L_list = [L_qubit, L_cavity]
    H_interaction = g * (np.kron(np.kron(sigma_plus, sigma_minus), a) +
                         np.kron(np.kron(sigma_minus, sigma_plus), a_dagger))
    H_base = H_drive_qubit + H_interaction
    return H_base, L_list


def get_initial_state():
    """Construct the initial Bell state (embedded in the full system space)."""
    psi_bell = (np.kron([1, 0], [0, 1]) + np.kron([0, 1], [1, 0])) / np.sqrt(2)
    rho_bell = np.outer(psi_bell, psi_bell.conj())
    rho_init = np.zeros((dim_system, dim_system), dtype=complex)
    rho_init[:4, :4] = rho_bell
    return np.concatenate([rho_init.real.flatten(), rho_init.imag.flatten()])


# ============================================================================
# 4) Simulation Functions
# ============================================================================

def lindblad_rhs(t, rho_flat, H_base, L_list, beta_max, tau_f):
    """
    Master equation with continuous adaptive feedback.
    Feedback is activated for times t >= tau_f.
    """
    half = len(rho_flat) // 2
    rho_real = rho_flat[:half].reshape(dim_system, dim_system)
    rho_imag = rho_flat[half:].reshape(dim_system, dim_system)
    rho = rho_real + 1j * rho_imag

    # Unitary evolution and dissipators
    drho_dt = -1j / hbar * (H_base @ rho - rho @ H_base)
    for L in L_list:
        drho_dt += (L @ rho @ L.conj().T -
                    0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L))

    # Adaptive feedback applied after delay tau_f
    if t >= tau_f:
        rho_qubits = partial_trace_cavity(rho, dim_qubit, n_max)
        entanglement = compute_concurrence(rho_qubits)
        beta = adaptive_feedback(beta_max, entanglement)
        H_feedback = beta * (np.kron(np.kron(sigma_x, sigma_x), np.eye(n_max)) +
                             np.kron(np.eye(4), (a + a_dagger)))
        drho_dt += -1j * (H_feedback @ rho - rho @ H_feedback)

    return np.concatenate([drho_dt.real.flatten(), drho_dt.imag.flatten()])


def simulate_system(g, beta_max, tau_f, time_points):
    """
    Run the simulation for given parameters.

    Returns the solver object and the base Hamiltonian.
    """
    H_base, L_list = get_system_operators(g)
    rho_init_flat = get_initial_state()
    sol = solve_ivp(
        lindblad_rhs,
        t_span=[0, total_time],
        y0=rho_init_flat,
        t_eval=time_points,
        args=(H_base, L_list, beta_max, tau_f),
        method='RK45',
        atol=1e-8,
        rtol=1e-6
    )
    return sol, H_base


def process_solution(sol, H_base):
    """
    Process the simulation result to compute time series for concurrence,
    ergotropy, cavity occupation, and qubit population.
    """
    concurrence_ts = []
    ergotropy_ts = []
    cavity_occ_ts = []
    qubit_pop_ts = []
    sigma_plus_sigma_minus_full = np.kron(np.kron(sigma_plus @ sigma_minus, np.eye(2)), np.eye(n_max))
    for idx in range(len(sol.t)):
        rho_t = (sol.y[:dim_system ** 2, idx] + 1j * sol.y[dim_system ** 2:, idx]).reshape(dim_system, dim_system)
        rho_t = enforce_positivity(rho_t)
        rho_2q = partial_trace_cavity(rho_t, dim_qubit, n_max)
        concurrence_ts.append(compute_concurrence(rho_2q))
        ergotropy_ts.append(compute_ergotropy(rho_t, H_base))
        a_dagger_a_full = np.kron(np.eye(4), a_dagger @ a)
        cavity_occ_ts.append(compute_cavity_occupation(rho_t, a_dagger_a_full))
        qubit_pop_ts.append(compute_qubit_population(rho_t, sigma_plus_sigma_minus_full))
    return concurrence_ts, ergotropy_ts, cavity_occ_ts, qubit_pop_ts


# ============================================================================
# 5) Extended Visualization Functions (Heatmaps)
# ============================================================================

def run_heatmap_simulations(tau_f_values, coupling_strengths, feedback_strengths, time_points):
    param_labels = []
    concurrence_results = []
    ergotropy_results = []
    for tau in tau_f_values:
        for g in coupling_strengths:
            for beta in feedback_strengths:
                sol, H_base = simulate_system(g, beta, tau, time_points)
                concurrence, ergotropy, _, _ = process_solution(sol, H_base)
                label = f"g={g * GHz_to_MHz:.0f} MHz, β={beta:.2f}, τ={tau}"
                param_labels.append(label)
                concurrence_results.append(concurrence)
                ergotropy_results.append(ergotropy)
    return param_labels, np.array(concurrence_results), np.array(ergotropy_results)


def plot_heatmaps(param_labels, concurrence_matrix, ergotropy_matrix):
    fig, axs = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    im0 = axs[0].imshow(concurrence_matrix, aspect='auto', extent=[0, total_time, 0, len(param_labels)], origin='lower')
    axs[0].set_yticks(np.arange(len(param_labels)))
    axs[0].set_yticklabels(param_labels)
    axs[0].set_title('Concurrence Over Time for Different Parameters')
    axs[0].set_ylabel('Parameter Combination')
    fig.colorbar(im0, ax=axs[0], label='Concurrence')

    im1 = axs[1].imshow(ergotropy_matrix, aspect='auto', extent=[0, total_time, 0, len(param_labels)], origin='lower')
    axs[1].set_yticks(np.arange(len(param_labels)))
    axs[1].set_yticklabels(param_labels)
    axs[1].set_title('Ergotropy Over Time for Different Parameters')
    axs[1].set_xlabel('Time')
    axs[1].set_ylabel('Parameter Combination')
    fig.colorbar(im1, ax=axs[1], label='Ergotropy')

    plt.tight_layout()
    plt.show()


def plot_time_series(t, data_series, ylabel, title, legends):
    plt.figure(figsize=(10, 6))
    for series, leg in zip(data_series, legends):
        plt.plot(t, series, label=leg)
    plt.xlabel('Time')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# ============================================================================
# 6) Main Execution
# ============================================================================

def main():
    # --- Time Series Simulations for Default tau_f ---
    simulation_results = []
    labels = []
    for g in coupling_strengths:
        for beta in feedback_strengths:
            sol, H_base = simulate_system(g, beta, tau_f_default, time_points)
            concurrence, ergotropy, cavity_occ, qubit_pop = process_solution(sol, H_base)
            lbl = f"g={g * GHz_to_MHz:.2f} MHz, β={beta:.2f}"
            simulation_results.append({
                't': sol.t,
                'concurrence': concurrence,
                'ergotropy': ergotropy,
                'cavity_occ': cavity_occ,
                'qubit_pop': qubit_pop
            })
            labels.append(lbl)

    # Plot each observable for each simulation
    for res, lbl in zip(simulation_results, labels):
        plot_time_series(res['t'], [res['concurrence']], 'Concurrence',
                         f'Concurrence Time Evolution: {lbl}', [lbl])
        plot_time_series(res['t'], [res['ergotropy']], 'Ergotropy',
                         f'Extractable Work (Ergotropy): {lbl}', [lbl])
        plot_time_series(res['t'], [res['cavity_occ']], 'Cavity Photon Number ⟨n⟩',
                         f'Cavity Occupation: {lbl}', [lbl])
        plot_time_series(res['t'], [res['qubit_pop'], res['ergotropy']], 'Expectation Values',
                         f'Qubit Excitation vs. Ergotropy: {lbl}', ['Qubit Excitation', 'Ergotropy'])

    # --- Extended Visualization: Heatmaps Across g, β, and τ_f ---
    tau_f_values = [0.5, 1.0, 2.0]
    param_labels, conc_mat, ergo_mat = run_heatmap_simulations(tau_f_values, coupling_strengths, feedback_strengths,
                                                               time_points)
    plot_heatmaps(param_labels, conc_mat, ergo_mat)


if __name__ == '__main__':
    main()
