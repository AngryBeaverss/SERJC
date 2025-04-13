import numpy as np
from observables import compute_concurrence

def adaptive_feedback(beta_max, entanglement):
    return min(beta_max * (1 - entanglement) * np.exp(-entanglement), 0.02)

def get_feedback_hamiltonian(mode, rho, params, t, tau_f, beta_max, rho_target=None):
    if t < tau_f or mode == 'none':
        return np.zeros_like(rho)

    beta = 0.0
    if mode == 'adaptive':
        rho_qubits = partial_trace_cavity(rho, 4, params['n_max'])
        entanglement = compute_concurrence(rho_qubits)
        beta = adaptive_feedback(beta_max, entanglement)
    elif mode == 'fixed':
        beta = beta_max
    elif mode == 'lyapunov' and rho_target is not None:
        commutator = rho_target @ rho - rho @ rho_target
        return 1j * 0.1 * (commutator - commutator.conj().T)

    H_feedback = beta * (
        np.kron(np.kron(params['sigma_x'], params['sigma_x']), np.eye(params['n_max'])) +
        np.kron(np.eye(4), params['a'] + params['a_dagger'])
    )
    return H_feedback

# Needed for internal use
def partial_trace_cavity(rho_full, n_qubits=4, n_cavity=15):
    rho_4d = rho_full.reshape(n_qubits, n_cavity, n_qubits, n_cavity)
    return np.trace(rho_4d, axis1=1, axis2=3)
