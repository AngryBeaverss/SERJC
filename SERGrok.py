import numpy as np
import pyopencl as cl
from matplotlib import pyplot as plt

##############################################################################
# 1) OPENCL SETUP (unchanged, assumed optimal)
##############################################################################
platform = cl.get_platforms()[0]
device = platform.get_devices()[0]
context = cl.Context([device])
queue = cl.CommandQueue(context)

with open('ser_lindblad.cl', 'r') as f:
    kernel_code = f.read()
program = cl.Program(context, kernel_code).build()

##############################################################################
# 2) SIMULATION PARAMETERS
##############################################################################
dt = 0.001
total_time = 100.0
num_steps = int(total_time / dt) + 1  # Include t=0
time_axis = np.linspace(0, total_time, num_steps)  # Precompute once

dimensions = [2, 4, 8, 16]
results_by_dimension = {}
mf = cl.mem_flags  # Predefine memory flags

for DSIZE in dimensions:
    print(f"\nRunning simulation for dimension: {DSIZE}x{DSIZE}")

    ##########################################################################
    # 3) SYSTEM MATRICES
    ##########################################################################
    # Use float64 for real parts where possible, complex128 only where needed
    A = np.random.randn(DSIZE, DSIZE) + 1j * np.random.randn(DSIZE, DSIZE)
    rho_init = A @ A.conj().T
    rho_init /= np.trace(rho_init)  # In-place normalization

    H_init = np.random.rand(DSIZE, DSIZE) + 1j * np.random.rand(DSIZE, DSIZE)
    H_init = 0.5 * (H_init + H_init.conj().T)  # Hermitian, in-place scaling

    L_init = np.random.rand(DSIZE, DSIZE) + 1j * np.random.rand(DSIZE, DSIZE)
    L_init = 0.5 * (L_init + L_init.conj().T)

    I_init = np.eye(DSIZE, dtype=np.complex128)

    ##########################################################################
    # 4) CREATE OPENCL BUFFERS
    ##########################################################################
    rho_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=rho_init)
    H_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=H_init)
    L_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=L_init)
    I_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=I_init)

    # Preallocate arrays with float64 for real-valued quantities
    purity_vals = np.zeros(num_steps, dtype=np.float64)
    entropy_vals = np.zeros(num_steps, dtype=np.float64)
    coherence_vals = np.zeros(num_steps, dtype=np.float64)
    energy_derivative_vals = np.zeros(num_steps - 1, dtype=np.float64)

    ##########################################################################
    # 5) HELPER FUNCTIONS (Optimized)
    ##########################################################################
    log2_DSIZE = np.log2(DSIZE)  # Precompute for entropy normalization

    def compute_quantities(rho_m):
        purity = np.real(np.trace(rho_m @ rho_m))
        w = np.linalg.eigvalsh(rho_m)
        w_pos = w[w > 1e-12]
        entropy = -np.sum(w_pos * np.log2(w_pos)) if w_pos.size > 0 else 0.0
        coherence = np.abs(rho_m).sum() - np.abs(rho_m).trace()
        return purity, entropy, coherence

    def compute_energy_derivative(rho_prev, rho_curr, H):
        E_prev = np.real(np.trace(rho_prev @ H))
        E_curr = np.real(np.trace(rho_curr @ H))
        return (E_curr - E_prev) / dt

    ##########################################################################
    # 6) INITIALIZATION
    ##########################################################################
    rho_host = rho_init.copy()  # Keep 2D, avoid flattening/reshaping
    p0, e0, c0 = compute_quantities(rho_host)
    purity_vals[0], entropy_vals[0], coherence_vals[0] = p0, e0, np.real(c0)
    hbar = 1.0

    ##########################################################################
    # 7) MAIN LOOP (Optimized)
    ##########################################################################
    rho_prev = np.empty_like(rho_host)  # Preallocate once
    beta_0 = 1.0
    for step in range(1, num_steps):
        rho_prev[...] = rho_host  # In-place copy

        coherence_prev = coherence_vals[step - 1]
        gamma = 0.3 * np.exp(np.clip(-2.0 * (1.0 - coherence_prev), -50, 50))

        norm_entropy = entropy_vals[step - 1] / log2_DSIZE if step > 1 else 0.0
        beta = beta_0 * (1 - np.exp(norm_entropy))
        beta = min(beta, 2.0)

        # Asynchronous kernel execution
        event = program.ser_lindblad(queue, (DSIZE,), None, rho_buf, H_buf, L_buf, I_buf,
                                     np.float64(hbar), np.float64(gamma), np.float64(beta), np.float64(dt))
        cl.enqueue_copy(queue, rho_host, rho_buf, wait_for=[event], is_blocking=False).wait()
        rho_host[...] = np.nan_to_num(rho_host, nan=1e-6, posinf=1e-6, neginf=1e-6)

        p, e, c = compute_quantities(rho_host)
        purity_vals[step] = p
        entropy_vals[step] = e
        coherence_vals[step] = np.real(c)

        energy_derivative_vals[step - 1] = compute_energy_derivative(rho_prev, rho_host, H_init)

    ##########################################################################
    # 8) STORE RESULTS
    ##########################################################################
    cumulative_energy = np.cumsum(energy_derivative_vals) * dt
    results_by_dimension[DSIZE] = {
        'purity': purity_vals,
        'entropy': entropy_vals,
        'coherence': coherence_vals,
        'cumulative_energy': cumulative_energy
    }

    print(f"Final Purity={purity_vals[-1]:.6f}, Entropy={entropy_vals[-1]:.6f}, Coherence={coherence_vals[-1]:.6f}")

##############################################################################
# 9) PLOTTING RESULTS (Optimized)
##############################################################################
fig, axes = plt.subplots(2, 2, figsize=(15, 10))  # Single figure with subplots
metrics = ['purity', 'entropy', 'coherence', 'cumulative_energy']
for ax, metric in zip(axes.flat, metrics):
    for DSIZE in dimensions:
        data = results_by_dimension[DSIZE][metric]
        t_axis = time_axis if metric != 'cumulative_energy' else time_axis[:-1]
        ax.plot(t_axis, data, label=f'{DSIZE}x{DSIZE}')
    ax.set_xlabel('Time')
    ax.set_ylabel(metric.capitalize())
    ax.set_title(f'{metric.capitalize()} across Dimensions')
    ax.grid(True)
    ax.legend()

plt.tight_layout()
plt.savefig("SER_metrics_scaling.png", dpi=300)
plt.show()