import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

##############################################################################
# 1) Constants and System Setup
##############################################################################
hbar = 1.0
GHz_to_MHz = 1000

# Time settings
total_time = 200.0
time_points = np.linspace(0, total_time, 400)

# Truncation
n_max = 15  # Fock states for the cavity

# Experimental-like parameters
omega_qubit_real = 5.0
omega_cavity_real = 5.0
drive_strength_real = 10 / GHz_to_MHz
gamma_spont_real = 1 / GHz_to_MHz
kappa_real = 0.1 / GHz_to_MHz
beta_max = 0.02

# Feedback levels to test
feedback_strengths = [0.0, 0.5]  # Different β_max values

# Coupling strengths to test
coupling_strengths = [20 / GHz_to_MHz, 50 / GHz_to_MHz]  # Weak, strong, ultra-strong

tau_f = 1.0  # Feedback delay

# Pauli matrices
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
sigma_plus = sigma_minus.T

# a, a_dagger for the cavity
a = np.diag(np.sqrt(np.arange(1, n_max)), 1)
a_dagger = a.T

# System dimension
dim_system = 4 * n_max


##############################################################################
# 2) Helper Functions
##############################################################################

def enforce_positivity(rho):
    """ Fix small numerical negativity in the density matrix. """
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.clip(eigvals, 1e-10, None)  # Clip negative values
    rho_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
    return rho_fixed / np.trace(rho_fixed)


def adaptive_feedback(beta_max, entanglement):
    """ Adaptive feedback scaling based on concurrence. """
    return min(beta_max * (1 - entanglement) * np.exp(-entanglement), 0.02)


def partial_trace_cavity(rho_full, n_qubits=4, n_cavity=n_max):
    """ Trace out the cavity, returning a two-qubit density matrix. """
    rho_4d = rho_full.reshape(n_qubits, n_cavity, n_qubits, n_cavity)
    return np.trace(rho_4d, axis1=1, axis2=3)


def compute_concurrence(rho_2q):
    """ Compute the Wootters concurrence for a 4x4 two-qubit density matrix. """
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_yy = np.kron(sigma_y, sigma_y)
    rho_star = np.conjugate(rho_2q)
    rho_tilde = sigma_yy @ rho_star @ sigma_yy
    product = rho_2q @ rho_tilde
    eigenvals = np.linalg.eigvals(product)
    eigenvals = np.sort(eigenvals)[::-1].real
    sqrt_vals = np.sqrt(np.clip(eigenvals, 0, None))
    return max(0.0, sqrt_vals[0] - sqrt_vals[1] - sqrt_vals[2] - sqrt_vals[3])


def compute_ergotropy(rho, H):
    """
    Computes the extractable work (ergotropy) from the density matrix ρ and Hamiltonian H.
    """
    # System energy
    energy = np.trace(rho @ H).real

    # Diagonalize ρ to find passive state
    eigvals_rho, eigvecs_rho = np.linalg.eigh(rho)

    # Sort eigenvalues in descending order
    sorted_indices = np.argsort(eigvals_rho)[::-1]
    eigvals_rho = eigvals_rho[sorted_indices]
    eigvecs_rho = eigvecs_rho[:, sorted_indices]

    # Diagonalize H
    eigvals_H, eigvecs_H = np.linalg.eigh(H)

    # Sort eigenvalues in ascending order (passive state ordering)
    sorted_indices_H = np.argsort(eigvals_H)
    eigvals_H = eigvals_H[sorted_indices_H]
    eigvecs_H = eigvecs_H[:, sorted_indices_H]

    # Construct passive state
    rho_passive = sum(
        eigvals_rho[i] * np.outer(eigvecs_H[:, i], eigvecs_H[:, i].conj()) for i in range(len(eigvals_rho)))

    # Passive energy
    passive_energy = np.trace(rho_passive @ H).real

    # Ergotropy = Maximum work extractable
    return energy - passive_energy

def compute_cavity_occupation(rho, a_dagger_a_full):
    """
    Computes the expectation value of the cavity photon number operator,
    correctly embedded in the full Hilbert space.
    """
    return np.real(np.trace(rho @ a_dagger_a_full))

def compute_qubit_population(rho, sigma_plus_sigma_minus_full):
    """
    Computes the expectation value of the qubit excitation number operator.
    """
    return np.real(np.trace(rho @ sigma_plus_sigma_minus_full))


##############################################################################
# 3) Main Loop Over Different Coupling Strengths and Feedback Levels
##############################################################################
plt.figure(figsize=(10, 6))

for g in coupling_strengths:
    for beta_max in feedback_strengths:
        # Define Hamiltonian and Lindblad Operators
        H_drive_qubit = drive_strength_real * np.kron(np.kron(sigma_x, np.eye(2)), np.eye(n_max))
        L_qubit_real = np.sqrt(gamma_spont_real) * np.kron(np.kron(sigma_minus, np.eye(2)), np.eye(n_max))
        L_cavity_real = np.sqrt(kappa_real) * np.kron(np.eye(4), a)
        L_list_real = [L_qubit_real, L_cavity_real]

        # Jaynes-Cummings interaction
        H_interaction = g * (
                np.kron(np.kron(sigma_plus, sigma_minus), a)
                + np.kron(np.kron(sigma_minus, sigma_plus), a_dagger)
        )

        H_base = H_drive_qubit + H_interaction

        # Define the initial Bell state in the qubit subspace
        psi_bell = (np.kron([1, 0], [0, 1]) + np.kron([0, 1], [1, 0])) / np.sqrt(2)
        rho_bell = np.outer(psi_bell, psi_bell.conj())

        # Expand to match full system dimension
        rho_init = np.zeros((dim_system, dim_system), dtype=complex)
        rho_init[:4, :4] = rho_bell  # Place Bell state in qubit subspace
        rho_init_flat = np.concatenate([rho_init.real.flatten(), rho_init.imag.flatten()])


        # Modify the feedback logic to continuously apply adaptive feedback after delay
        def lindblad_rhs_continuous_feedback(t, rho_flat, H_base, L_list, beta_max):
            half_len = len(rho_flat) // 2
            rho_real = rho_flat[:half_len].reshape(dim_system, dim_system)
            rho_imag = rho_flat[half_len:].reshape(dim_system, dim_system)
            rho = rho_real + 1j * rho_imag

            drho_dt = -1j / hbar * (H_base @ rho - rho @ H_base)
            for L in L_list:
                lindblad_term = L @ rho @ L.conj().T - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L)
                drho_dt += lindblad_term

            # Apply adaptive feedback *continuously* after tau_f
            if t >= tau_f:
                rho_qubits = partial_trace_cavity(rho, 4, n_max)
                entanglement = compute_concurrence(rho_qubits)
                beta = adaptive_feedback(beta_max, entanglement)

                H_feedback = beta * (
                        np.kron(np.kron(sigma_x, sigma_x), np.eye(n_max)) +
                        np.kron(np.eye(4), (a + a_dagger))
                )
                drho_dt += -1j * (H_feedback @ rho - rho @ H_feedback)

            return np.concatenate([drho_dt.real.flatten(), drho_dt.imag.flatten()])


        # Solve with continuous feedback applied after tau_f
        sol_cont = solve_ivp(
            lindblad_rhs_continuous_feedback,
            t_span=[0, total_time],
            y0=rho_init_flat,
            t_eval=time_points,
            args=(H_base, L_list_real, beta_max),
            method='RK45',
            atol=1e-8, rtol=1e-6
        )


        ergotropy_over_time = []
        cavity_photon_occupation = []
        correlation_data = []
        qubit_population = []
        ergotropy_cont = []
        concurrence_cont = []

        sigma_plus_sigma_minus_full = np.kron(np.kron(sigma_plus @ sigma_minus, np.eye(2)), np.eye(n_max))

        # Compute concurrence over time
        concurrence_over_time = []

        for idx in range(len(sol_cont.t)):
            rho_t = (sol_cont.y[:dim_system ** 2, idx] + 1j * sol_cont.y[dim_system ** 2:, idx]).reshape(dim_system, dim_system)
            rho_t = enforce_positivity(rho_t)
            rho_2q = partial_trace_cavity(rho_t, 4, n_max)
            concurrence_over_time.append(compute_concurrence(rho_2q))

            # Compute ergotropy
            W_ex = compute_ergotropy(rho_t, H_base)
            ergotropy_over_time.append(W_ex)

             # Corrected cavity photon number calculation
            a_dagger_a_full = np.kron(np.eye(4), a_dagger @ a)
            n_cav = compute_cavity_occupation(rho_t, a_dagger_a_full)

            cavity_photon_occupation.append(n_cav)

            # Compute concurrence
            rho_2q = partial_trace_cavity(rho_t, 4, n_max)
            concurrence_val = compute_concurrence(rho_2q)
            concurrence_cont.append(compute_concurrence(rho_2q))
            ergotropy_cont.append(compute_ergotropy(rho_t, H_base))

            # Store both
            correlation_data.append((concurrence_val, W_ex))

            # Compute qubit population
            p_excited = compute_qubit_population(rho_t, sigma_plus_sigma_minus_full)
            qubit_population.append(p_excited)


plt.plot(sol_cont.t, concurrence_over_time, label=f'g = {g * GHz_to_MHz:.2f} MHz, β = {beta_max:.2f}')
plt.xlabel('Time')
plt.ylabel('Concurrence')
plt.title('Effect of Coupling Strength and Feedback on Concurrence')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(sol_cont.t, ergotropy_over_time, label=f'g = {g * GHz_to_MHz:.2f} MHz, β = {beta_max:.2f}')
plt.xlabel('Time')
plt.ylabel('Extractable Work (Ergotropy)')
plt.title('Evolution of Extractable Work in Two-Qubit System')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(sol_cont.t, cavity_photon_occupation, label=f'g = {g * GHz_to_MHz:.2f} MHz, β = {beta_max:.2f}')
plt.xlabel('Time')
plt.ylabel('Cavity Photon Number ⟨n⟩')
plt.title('Evolution of Cavity Photon Occupation')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

concurrence_vals, ergotropy_vals = zip(*correlation_data)

plt.figure(figsize=(10, 6))
plt.scatter(concurrence_vals, ergotropy_vals, alpha=0.7)
plt.xlabel('Concurrence')
plt.ylabel('Extractable Work (Ergotropy)')
plt.title('Correlation Between Concurrence and Ergotropy')
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(sol_cont.t, qubit_population, label='Qubit Excitation Probability')
plt.plot(sol_cont.t, ergotropy_over_time, label='Extractable Work (Ergotropy)', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Expectation Values')
plt.title('Qubit Excitation vs. Extractable Work')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot updated results
plt.figure(figsize=(10, 5))
plt.plot(sol_cont.t, ergotropy_cont, label="Ergotropy (Continuous Feedback)")
plt.plot(sol_cont.t, concurrence_cont, label="Concurrence (Entanglement)", linestyle='dashed')
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Continuous SER Feedback: Extractable Work vs. Concurrence")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()









