import numpy as np
from operators import load_operators

n_max = 15  # make sure this matches your settings

params = {
    'hbar': 1.0,
    'GHz_to_MHz': 1000,
    'total_time': 200.0,
    'num_time_points': 400,
    'n_max': n_max,
    'omega_qubit_real': 5.0,
    'omega_cavity_real': 5.0,
    'drive_strength_real': 10 / 1000,
    'gamma_spont_real': 1 / 1000,
    'kappa_real': 0.1 / 1000,
    'beta_max': 0.02,
    'tau_f': 1.0,
    'feedback_strengths': [0.0, 0.5],
    'coupling_strengths': [20 / 1000, 50 / 1000],
}

params.update(load_operators(n_max))
# Pauli matrices
params['sigma_x'] = np.array([[0, 1], [1, 0]], dtype=complex)
params['sigma_y'] = np.array([[0, -1j], [1j, 0]], dtype=complex)
params['sigma_z'] = np.array([[1, 0], [0, -1]], dtype=complex)

# Ladder operators
params['sigma_minus'] = np.array([[0, 1], [0, 0]], dtype=complex)
params['sigma_plus'] = params['sigma_minus'].T

# Cavity ladder
params['a'] = np.diag(np.sqrt(np.arange(1, n_max)), 1)
params['a_dagger'] = params['a'].T

