import numpy as np
from scipy.integrate import solve_ivp
from utils import partial_trace_cavity
from observables import compute_concurrence
from feedback import adaptive_feedback

def run_simulation(params, time_points, rho_init_flat, dim_system, H_base, L_list_real):

    def lindblad_rhs(t, rho_flat, H_base, L_list_real, beta_max, tau_f, dim_system, params):
        rho_flat = np.asarray(rho_flat)
        half_len = len(rho_flat) // 2
        rho_real = rho_flat[:half_len].reshape(dim_system, dim_system)
        rho_imag = rho_flat[half_len:].reshape(dim_system, dim_system)
        rho = rho_real + 1j * rho_imag

        drho_dt = -1j * (H_base @ rho - rho @ H_base)

        for L in L_list_real:
            drho_dt += L @ rho @ L.conj().T - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L)

        if t >= tau_f and params['feedback_mode'] != 'none':
            if params['feedback_mode'] == 'adaptive':
                rho_qubits = partial_trace_cavity(rho, 4, params['n_max'])
                entanglement = compute_concurrence(rho_qubits)
                beta = adaptive_feedback(beta_max, entanglement)
            elif params['feedback_mode'] == 'fixed':
                beta = beta_max
            else:
                beta = 0.0  # fallback

            H_feedback = beta * (
                    np.kron(np.kron(params['sigma_x'], params['sigma_x']), np.eye(params['n_max'])) +
                    np.kron(np.eye(4), params['a'] + params['a_dagger'])
            )
            drho_dt += -1j * (H_feedback @ rho - rho @ H_feedback)

        # Optional debug:
        if drho_dt.ndim != 2:
            print("⚠️ Warning: drho_dt is not a matrix!", drho_dt.shape, type(drho_dt))

        return np.concatenate([np.real(drho_dt).flatten(), np.imag(drho_dt).flatten()])

    solution = solve_ivp(
        lindblad_rhs,
        t_span=[0, params['total_time']],
        y0=rho_init_flat,
        t_eval=time_points,
        method='RK45',
        atol=1e-8,
        rtol=1e-6,
        args=(H_base, L_list_real, params['beta_max'], params['tau_f'], dim_system, params)
    )

    return solution
