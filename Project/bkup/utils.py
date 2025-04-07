import numpy as np

def enforce_positivity(rho):
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.clip(eigvals, 1e-10, None)
    rho_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
    return rho_fixed / np.trace(rho_fixed)

def partial_trace_cavity(rho_full, n_qubits=4, n_cavity=15):
    rho_4d = rho_full.reshape(n_qubits, n_cavity, n_qubits, n_cavity)
    return np.trace(rho_4d, axis1=1, axis2=3)

def compute_cavity_occupation(rho, a_dagger_a_full):
    return np.real(np.trace(rho @ a_dagger_a_full))
