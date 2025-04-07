import numpy as np

def compute_concurrence(rho_2q):
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
    energy = np.trace(rho @ H).real
    eigvals_rho, eigvecs_rho = np.linalg.eigh(rho)
    sorted_indices = np.argsort(eigvals_rho)[::-1]
    eigvals_rho = eigvals_rho[sorted_indices]
    eigvecs_rho = eigvecs_rho[:, sorted_indices]
    eigvals_H, eigvecs_H = np.linalg.eigh(H)
    sorted_indices_H = np.argsort(eigvals_H)
    eigvals_H = eigvals_H[sorted_indices_H]
    eigvecs_H = eigvecs_H[:, sorted_indices_H]
    rho_passive = sum(eigvals_rho[i] * np.outer(eigvecs_H[:, i], eigvecs_H[:, i].conj()) for i in range(len(eigvals_rho)))
    passive_energy = np.trace(rho_passive @ H).real
    return energy - passive_energy

def plot_ergotropy_difference(ax, t, erg_adaptive, erg_fixed, label_prefix="Diff"):
    diff = erg_adaptive - erg_fixed
    ax.plot(t, diff, label=f'{label_prefix} (adaptive - fixed) Ergotropy', linestyle='-.')
    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("SER Results")
    ax.grid(True)


def plot_results(ax, t, concurrence, ergotropy, photons, feedback_mode, coupling, show_derivative=True):
    label_prefix = f"{feedback_mode} | {coupling} MHz"
    ax.plot(t, concurrence, label=f'{label_prefix} - Concurrence')
    ax.plot(t, ergotropy, label=f'{label_prefix} - Ergotropy', linestyle='--')

    if show_derivative:
        t_diff = np.diff(t)
        erg_diff = np.diff(ergotropy)
        dE_dt = erg_diff / t_diff
        t_midpoints = 0.5 * (t[:-1] + t[1:])
        ax.plot(t_midpoints, dE_dt, label=f'{label_prefix} - dErgotropy/dt', linestyle='dashdot')

    ax.plot(t, photons, label=f'{label_prefix} - Photons', linestyle=':')
    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("SER Results")
    ax.grid(True)
    ax.legend()