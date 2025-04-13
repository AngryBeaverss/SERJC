import numpy as np

def enhanced_load_operators(n_max=15):
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
    sigma_plus = sigma_minus.T

    a = np.diag(np.sqrt(np.arange(1, n_max)), 1)
    a_dagger = a.T

    return {
        'sigma_x': sigma_x,
        'sigma_y': sigma_y,
        'sigma_z': sigma_z,
        'sigma_minus': sigma_minus,
        'sigma_plus': sigma_plus,
        'a': a,
        'a_dagger': a_dagger,
    }

def load_operators(n_max=15):
    sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
    sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
    sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)
    sigma_plus = sigma_minus.T

    a = np.diag(np.sqrt(np.arange(1, n_max)), 1)
    a_dagger = a.T

    return {
        'sigma_x': sigma_x,
        'sigma_y': sigma_y,
        'sigma_z': sigma_z,
        'sigma_minus': sigma_minus,
        'sigma_plus': sigma_plus,
        'a': a,
        'a_dagger': a_dagger,
    }