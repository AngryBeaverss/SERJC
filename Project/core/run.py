print("run.py has started execution.")  # Line 1
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))  # ✅ Force script's own dir into sys.path
import argparse
from simulation import run_simulation
from observables import compute_concurrence, compute_ergotropy
from utils import enforce_positivity, partial_trace_cavity, compute_cavity_occupation
from params import params
import numpy as np
import matplotlib.pyplot as plt
from observables import plot_results


def initialize_state(params):
    print("Initializing state...")
    psi_bell = (np.kron([1,0],[0,1]) + np.kron([0,1],[1,0])) / np.sqrt(2)
    rho_bell = np.outer(psi_bell, psi_bell.conj())
    rho_init = np.zeros((4 * params['n_max'], 4 * params['n_max']), dtype=complex)
    rho_init[:4, :4] = rho_bell
    return np.concatenate([rho_init.real.flatten(), rho_init.imag.flatten()])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--coupling', type=float, default=20.0)
    parser.add_argument('--feedback_mode', type=str, default='adaptive')  # <-- NEW
    parser.add_argument('--drive_strength_real', type=float)
    parser.add_argument('--gamma_spont_real', type=float)
    args = parser.parse_args()

    for k, v in vars(args).items():
        if v is not None and k in params:
            print(f"[PARAM OVERRIDE] {k} = {v}")
            params[k] = v

    coupling_strength = args.coupling / params['GHz_to_MHz']  # convert to GHz
    params['coupling_strength'] = args.coupling / params['GHz_to_MHz']
    params['feedback_mode'] = args.feedback_mode  # <-- NEW

    time_points = np.linspace(0, params['total_time'], params['num_time_points'])
    rho_init_flat = initialize_state(params)
    dim_system = 4 * params['n_max']
    a_dagger_a_full = np.kron(np.eye(4), params['a_dagger'] @ params['a'])

    # Hamiltonian setup (now using dynamic coupling strength)
    print("Building Hamiltonian...")
    H_drive_qubit = params['drive_strength_real'] * np.kron(np.kron(params['sigma_x'], np.eye(2)), np.eye(params['n_max']))
    H_interaction = coupling_strength * (
        np.kron(np.kron(params['sigma_plus'], params['sigma_minus']), params['a']) +
        np.kron(np.kron(params['sigma_minus'], params['sigma_plus']), params['a_dagger'])
    )
    H_base = H_drive_qubit + H_interaction

    # Lindblad operators
    L_qubit_real = np.sqrt(params['gamma_spont_real']) * np.kron(np.kron(params['sigma_minus'], np.eye(2)), np.eye(params['n_max']))
    L_cavity_real = np.sqrt(params['kappa_real']) * np.kron(np.eye(4), params['a'])
    L_list_real = [L_qubit_real, L_cavity_real]

    # Run the simulation
    print("Running simulation...")
    solution = run_simulation(params, time_points, rho_init_flat, dim_system, H_base, L_list_real)


    # Observables
    concurrence_over_time = []
    ergotropy_over_time = []
    photon_occupation = []  # inside your for-loop

    for idx in range(len(solution.t)):
        rho_t = (solution.y[:dim_system**2, idx] + 1j * solution.y[dim_system**2:, idx]).reshape(dim_system, dim_system)
        rho_t = enforce_positivity(rho_t)
        rho_2q = partial_trace_cavity(rho_t, 4, params['n_max'])
        concurrence_over_time.append(compute_concurrence(rho_2q))
        ergotropy_over_time.append(compute_ergotropy(rho_t, H_base))
        photon_occupation.append(compute_cavity_occupation(rho_t, a_dagger_a_full))

    print("Simulation finished. Computing observables...")
    np.savez("results.npz",
             time=solution.t,
             concurrence=concurrence_over_time,
             ergotropy=ergotropy_over_time,
             photons=photon_occupation)
    print("Saving results...")

    fig, ax = plt.subplots(figsize=(10, 5))
    plot_results(ax, solution.t, concurrence_over_time, ergotropy_over_time, photon_occupation, params['feedback_mode'],
                 args.coupling)
    plt.show()