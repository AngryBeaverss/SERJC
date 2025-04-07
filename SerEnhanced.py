import numpy as np
import pyopencl as cl
from matplotlib import pyplot as plt

##############################################################################
# 1) OPENCL SETUP
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
dt = .001
total_time = 100.0
num_steps = int(total_time / dt)
time_axis = np.linspace(0, total_time, num_steps + 1)

dimensions = [2, 4, 8, 16]  # System sizes to test
results_by_dimension = {}

for DSIZE in dimensions:
    print(f"\nRunning simulation for dimension: {DSIZE}x{DSIZE}")

    ##############################################################################
    # 3) SYSTEM MATRICES (Dynamic DSIZE)
    ##############################################################################
    A = np.random.randn(DSIZE, DSIZE) + 1j * np.random.randn(DSIZE, DSIZE)
    rho_init = A @ A.conj().T  # Ensure positive semi-definiteness
    rho_init /= np.trace(rho_init)  # Normalize trace = 1

    H_init = np.random.rand(DSIZE, DSIZE) + 1j * np.random.rand(DSIZE, DSIZE)
    H_init = (H_init + H_init.conj().T) / 2  # Hermitian

    L_init = np.random.rand(DSIZE, DSIZE) + 1j * np.random.rand(DSIZE, DSIZE)
    L_init = (L_init + L_init.conj().T) / 2  # Hermitian

    I_init = np.eye(DSIZE, dtype=np.complex128)

    ##############################################################################
    # 4) CREATE OPENCL BUFFERS
    ##############################################################################
    mf = cl.mem_flags
    rho_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=rho_init.flatten())
    H_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=H_init.flatten())
    L_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=L_init.flatten())
    I_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=I_init.flatten())

    purity_vals = np.zeros(num_steps + 1)
    entropy_vals = np.zeros(num_steps + 1)
    coherence_vals = np.zeros(num_steps + 1)
    trace_vals = np.zeros(num_steps + 1)
    energy_derivative_vals = np.zeros(num_steps)

    ##############################################################################
    # 5) HELPER FUNCTIONS
    ##############################################################################
    def compute_quantities(rho_1d):
        rho_m = rho_1d.reshape(DSIZE, DSIZE)
        purity = np.real(np.trace(rho_m @ rho_m))
        w = np.linalg.eigvalsh(rho_m)
        w_pos = w[w > 1e-12]
        entropy = -np.sum(w_pos * np.log2(w_pos)) if len(w_pos) > 0 else 0.0
        coherence = np.sum(np.abs(rho_m)) - np.trace(np.abs(rho_m))
        return purity, entropy, coherence

    def compute_energy_derivative(rho_prev, rho_curr, H, dt):
        E_prev = np.real(np.trace(rho_prev @ H))
        E_curr = np.real(np.trace(rho_curr @ H))
        return (E_curr - E_prev) / dt

    ##############################################################################
    # 6) INITIALIZATION (step = 0)
    ##############################################################################
    rho_host = np.empty_like(rho_init)
    cl.enqueue_copy(queue, rho_host, rho_buf).wait()
    rho_m = rho_host.reshape(DSIZE, DSIZE)

    p0, e0, c0 = compute_quantities(rho_host)
    purity_vals[0] = p0
    entropy_vals[0] = e0
    coherence_vals[0] = np.real(c0)
    hbar = 1

    ##############################################################################
    # 7) MAIN LOOP
    ##############################################################################
    for step in range(1, num_steps + 1):
        rho_prev = rho_host.reshape(DSIZE, DSIZE).copy()

        coherence_prev = coherence_vals[step - 1]
        gamma = 0.3 * np.exp(np.clip(-2.0 * (1.0 - coherence_prev), -50, 50))

        norm_entropy = entropy_vals[step - 1] / np.log2(DSIZE) if step > 1 else 0.0
        beta_0 = 1.0  # Baseline beta value
        beta = beta_0 * (1 - np.exp(norm_entropy))
        beta = min(beta, 2.0)

        program.ser_lindblad(queue, (DSIZE,), None, rho_buf, H_buf, L_buf, I_buf,np.float64(hbar), np.float64(gamma), np.float64(beta), np.float64(dt))

        cl.enqueue_copy(queue, rho_host, rho_buf).wait()
        rho_m = rho_host.reshape(DSIZE, DSIZE)
        rho_m = np.nan_to_num(rho_m, nan=1e-6, posinf=1e-6, neginf=1e-6)  # Replace NaNs/Infs
        rho_host = rho_m.flatten()

        p, e, c = compute_quantities(rho_host)
        purity_vals[step] = p
        entropy_vals[step] = e
        coherence_vals[step] = np.real(c)

        dE_dt = compute_energy_derivative(rho_prev, rho_m, H_init, dt)
        energy_derivative_vals[step - 1] = dE_dt

    ##############################################################################
    # 8) STORE RESULTS
    ##############################################################################
    cumulative_energy = np.cumsum(energy_derivative_vals) * dt
    results_by_dimension[DSIZE] = {
        'purity': purity_vals.copy(),
        'entropy': entropy_vals.copy(),
        'coherence': coherence_vals.copy(),
        'cumulative_energy': cumulative_energy.copy()
    }

    print(f"Final Purity={purity_vals[-1]:.6f}, Entropy={entropy_vals[-1]:.6f}, Coherence={coherence_vals[-1]:.6f}")

##############################################################################
# 9) PLOTTING RESULTS
##############################################################################
for metric in ['purity', 'entropy', 'coherence']:
    plt.figure(figsize=(10, 6))
    for DSIZE in dimensions:
        plt.plot(time_axis, results_by_dimension[DSIZE][metric], label=f'{DSIZE}x{DSIZE}')
    plt.xlabel('Time')
    plt.ylabel(metric.capitalize())
    plt.title(f'{metric.capitalize()} across Dimensions')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"SER_{metric}_scaling.png", dpi=300)
    plt.show()

# Plot cumulative energy across dimensions
plt.figure(figsize=(10, 6))
for DSIZE in dimensions:
    plt.plot(time_axis[:-1], results_by_dimension[DSIZE]['cumulative_energy'], label=f'{DSIZE}x{DSIZE}')
plt.xlabel('Time')
plt.ylabel('Cumulative Energy')
plt.title('Cumulative Energy Flow due to SER Feedback Across Dimensions')
plt.grid(True)
plt.legend()
plt.savefig("SER_cumulative_energy_scaling.png", dpi=300)
plt.show()
