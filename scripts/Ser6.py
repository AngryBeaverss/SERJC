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
dt = .001           # Smaller time-step => fewer numerical leaps
total_time = 100.0
num_steps = int(total_time / dt)


##############################################################################
# 3) SYSTEM MATRICES (2x2)
##############################################################################
# A simple mixed state (must be positive semidefinite and trace 1)
# # Create a random 4x4 complex matrix
# A = randn(4,4) + 1j*randn(4,4)
# # Make a positive semidefinite matrix: A A^\dagger
# rho_random = A @ A.conjugate().T
# # Normalize to trace=1
# rho_random /= np.trace(rho_random)
#
# rho_init = rho_random
# psi = np.random.randn(4) + 1j * np.random.randn(4)  # Random complex vector
# psi /= np.linalg.norm(psi)  # Normalize to unit length
# rho_init = np.outer(psi, psi.conj())  # Pure state density matrix
rho_init = np.array([
    [0.9, 0.1, 0.0, 0.0],
    [0.1, 0.1, 0.0, 0.0],
    [0.0, 0.0, 0.5, 0.5],
    [0.0, 0.0, 0.5, 0.5]
])

H_init = np.array([
    0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 1.0 + 0j,
    0.0 + 0j, 0.0 + 0j, 1.0 + 0j, 0.0 + 0j,
    0.0 + 0j, 1.0 + 0j, 0.0 + 0j, 0.0 + 0j,
    1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j
], dtype=np.complex128).reshape(4, 4)

L_init = np.array([
    0.0 + 0j, 1.0 + 0j, 0.0 + 0j, 0.0 + 0j,
    1.0 + 0j, 0.0 + 0j, 0.0 + 0j, 0.0 + 0j,
    0.0 + 0j, 0.0 + 0j, 0.0 + 0j, 1.0 + 0j,
    0.0 + 0j, 0.0 + 0j, 1.0 + 0j, 0.0 + 0j
], dtype=np.complex128).reshape(4, 4)

I_init = np.array([1.0 + 0j, 0.0 + 0j,
                   0.0 + 0j, 1.0 + 0j],
                  dtype=np.complex128)

##############################################################################
# 4) CREATE OPENCL BUFFERS
##############################################################################
mf = cl.mem_flags
rho_buf = cl.Buffer(context, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=rho_init.flatten())
H_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=H_init.flatten())
L_buf = cl.Buffer(context, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=L_init.flatten())
I_buf   = cl.Buffer(context, mf.READ_ONLY  | mf.COPY_HOST_PTR, hostbuf=I_init)

##############################################################################
# 5) ARRAYS TO STORE PURITY, ENTROPY, COHERENCE, TRACE
##############################################################################
purity_vals     = np.zeros(num_steps + 1)
entropy_vals    = np.zeros(num_steps + 1)
coherence_vals  = np.zeros(num_steps + 1)
trace_vals      = np.zeros(num_steps + 1)

##############################################################################
# 6) HELPER FUNCTIONS
##############################################################################
def compute_quantities(rho_1d):
    """Compute purity, entropy, coherence, trace from a 4x4 density matrix."""
    rho_m = rho_1d.reshape(4, 4)
    rho2 = rho_m @ rho_m
    purity = np.real(np.trace(rho2))

    # Eigenvalues for entropy
    w = np.linalg.eigvalsh(rho_m)
    w_pos = w[w > 1e-12]  # Ignore tiny negative drift
    entropy = -np.sum(w_pos * np.log2(w_pos)) if len(w_pos) > 0 else 0.0

    # Corrected Coherence: Sum of absolute off-diagonal elements
    coherence = np.sum(np.abs(rho_m)) - np.trace(np.abs(rho_m))

    trace = np.real(np.trace(rho_m))

    return purity, entropy, coherence, trace

def enforce_positivity(rho_m):
    """
    Project rho_m onto the positive semidefinite cone.
    1) Diagonalize
    2) Clamp negative eigenvalues to 0
    3) Reconstruct
    4) Normalize trace to 1
    """
    w, v = np.linalg.eigh(rho_m)
    w_clamped = np.clip(w, 0, None)
    rho_corrected = v @ np.diag(w_clamped) @ v.conj().T
    tr = np.trace(rho_corrected)
    if np.abs(tr) < 1e-15:
        # fallback if trace is near zero
        return np.eye(2, dtype=np.complex128) / 2
    return rho_corrected / tr

##############################################################################
# 7) INITIALIZATION (step = 0)
##############################################################################
# Read initial rho from GPU, positivity-project
rho_host = np.empty_like(rho_init)
cl.enqueue_copy(queue, rho_host, rho_buf).wait()

# rho_m = rho_host.reshape(2,2)
rho_m = rho_host.reshape(4,4)

rho_m = enforce_positivity(rho_m)
rho_host = rho_m.flatten()

# Compute initial quantities
p0, e0, c0, t0 = compute_quantities(rho_host)
purity_vals[0]    = p0
entropy_vals[0]   = e0
coherence_vals[0] = np.real(c0)
trace_vals[0]     = t0

# Write the positivity-corrected ρ back
cl.enqueue_copy(queue, rho_buf, rho_host).wait()

print(f"Step 0, Purity={p0:.4f}, Entropy={e0:.4f}, Coherence={c0:.4f}, Trace={t0:.4f}")

##############################################################################
# 8) MAIN LOOP
##############################################################################
hbar = 1.0

for step in range(1, num_steps + 1):

    # -- Define gamma, beta from the *previous* step's data (step-1) --
    coherence_prev = coherence_vals[step-1]
    # gamma = 0.3 e^{-2(1-coherence)}
    gamma = 0.3 * np.exp(-2.0 * (1.0 - coherence_prev))

    # Measure change in entropy from the previous step (only if step > 1)
    if step > 1:
        ent_change = abs(entropy_vals[step-1] - entropy_vals[step-2])
    else:
        ent_change = 0.0

    # Assume Hilbert space dimension d (for your simulation, d = 4)
    d = 4

    # Use the state quantities computed at the previous step:
    purity_current = purity_vals[step - 1]
    entropy_current = entropy_vals[step - 1]
    coherence_current = coherence_vals[step - 1]

    # Normalize entropy: maximum for a d-dimensional system is log2(d)
    norm_entropy = entropy_current / np.log2(d)
    # F_entropy = np.exp(-entropy_current)
    F_combined = 0.3 * (1 - purity_current) + 0.7 * coherence_current


    # Define baseline beta and entropy-based feedback
    beta_0 = 0.3  # Baseline beta value
    k_e = norm_entropy  # Coefficient for entropy feedback

    # Beta based ONLY on entropy change
    coherence_change = abs(coherence_vals[step] - coherence_vals[step - 1])
    # beta = beta_0 * (1 + coherence_change) * F_combined
    beta = beta_0 * (1 - np.exp(-coherence_prev))
    # beta = beta_0 * F_entropy
    beta = min(beta, 2.0)

    # -- Evolve one time-step via the kernel --
    program.ser_lindblad(
        queue, (4,), None,
        rho_buf, H_buf, L_buf, I_buf,
        np.float64(hbar), np.float64(gamma), np.float64(beta), np.float64(dt)
    )

    # -- Read the new ρ --
    cl.enqueue_copy(queue, rho_host, rho_buf).wait()
    rho_m = rho_host.reshape(4,4)

    # -- Positivity projection --
    rho_m = enforce_positivity(rho_m)
    rho_host = rho_m.flatten()

    # -- Compute and store updated quantities (this is step # step) --
    p, e, c, t = compute_quantities(rho_host)
    purity_vals[step]    = p
    entropy_vals[step]   = e
    coherence_vals[step] = np.real(c)
    trace_vals[step]     = t

    # -- Write corrected ρ back to GPU for next iteration --
    cl.enqueue_copy(queue, rho_buf, rho_host).wait()

    # -- Print occasionally --
    if step % 10000 == 0:
        print(f"Step {step}, Purity={p:.4f}, Entropy={e:.4f}, "
              f"Coherence={c:.4f}, Trace={t:.4f}, gamma={gamma:.4f}, beta={beta:.6f}")


##############################################################################
# 9) FINAL OUTPUT AND PLOTTING
##############################################################################
final_p = purity_vals[-1]
final_e = entropy_vals[-1]
final_c = coherence_vals[-1]
final_t = trace_vals[-1]

print("\n--- Final Values ---")
print(f"Purity = {final_p:.6f}")
print(f"Entropy = {final_e:.6f}")
print(f"Coherence = {final_c:.6f}")
print(f"Trace = {final_t:.6f}")

time_axis = np.linspace(0, total_time, num_steps + 1)
plt.figure(figsize=(10, 6))
plt.plot(time_axis, purity_vals,    label='Purity')
plt.plot(time_axis, entropy_vals,   label='Entropy')
plt.plot(time_axis, coherence_vals, label='Coherence')
plt.plot(time_axis, trace_vals,     label='Trace', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Quantity')
plt.legend()
plt.title('Evolution of Purity, Entropy, Coherence, and Trace (SER with Positivity Projection)')
plt.grid(True)
plt.show()

trace_var = np.max(trace_vals) - np.min(trace_vals)
print(f"Trace variation: {trace_var:.1e}")
