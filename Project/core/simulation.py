import numpy as np
from scipy.integrate import solve_ivp
from feedback import get_feedback_hamiltonian

def run_simulation(params, time_points, rho_init_flat, dim_system, H_base, L_list_real):
    def lindblad_rhs(t, rho_flat):
        rho_flat = np.asarray(rho_flat)
        half_len = len(rho_flat) // 2
        rho_real = rho_flat[:half_len].reshape(dim_system, dim_system)
        rho_imag = rho_flat[half_len:].reshape(dim_system, dim_system)
        rho = rho_real + 1j * rho_imag

        drho_dt = -1j * (H_base @ rho - rho @ H_base)

        for L in L_list_real:
            drho_dt += L @ rho @ L.conj().T - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L)

        H_fb = get_feedback_hamiltonian(
            mode=params['feedback_mode'],
            rho=rho,
            params=params,
            t=t,
            tau_f=params['tau_f'],
            beta_max=params['beta_max'],
            rho_target=None  # Optional: insert target state here if using Lyapunov
        )

        drho_dt += -1j * (H_fb @ rho - rho @ H_fb)

        return np.concatenate([np.real(drho_dt).flatten(), np.imag(drho_dt).flatten()])

    solution = solve_ivp(
        lindblad_rhs,
        t_span=[0, params['total_time']],
        y0=rho_init_flat,
        t_eval=time_points,
        method='RK45',
        atol=1e-8,
        rtol=1e-6
    )

    return solution