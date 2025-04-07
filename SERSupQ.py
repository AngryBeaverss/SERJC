import numpy as np
import matplotlib.pyplot as plt

# Simulation parameters
dt = 0.001  # Small timestep for numerical stability
total_time = 50.0  # Shorter total time for clear trends
num_steps = int(total_time / dt)
time_axis = np.linspace(0, total_time, num_steps + 1)

# System size
DSIZE = 2

# Define Hamiltonian and Lindblad operator
omega_0 = 1.0
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
H = 0.5 * omega_0 * sigma_z  # Hamiltonian

gamma_value = 0.02  # Moderate decay rate for clear contrast
L = np.array([[0, 0], [np.sqrt(gamma_value), 0]], dtype=np.complex128)  # Lindblad collapse operator
I = np.eye(DSIZE, dtype=np.complex128)

# Initial density matrix (pure superposition state)
rho_init = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)

# Function to compute purity, entropy, coherence, trace
def compute_quantities(rho):
    purity = np.real(np.trace(rho @ rho))
    eigvals = np.linalg.eigvalsh(rho)
    entropy = -np.sum(eigvals * np.log2(eigvals + 1e-10)) if np.any(eigvals > 0) else 0.0
    coherence = np.abs(rho[0, 1]) + np.abs(rho[1, 0])
    trace = np.real(np.trace(rho))
    return purity, entropy, coherence, trace

# Function to enforce positivity and normalize trace
def enforce_positivity(rho):
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.clip(eigvals, 0, None)  # Ensure no negative eigenvalues
    rho_fixed = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
    return rho_fixed / np.trace(rho_fixed)  # Normalize trace to 1

# Function for Lindblad evolution with and without SER
def ser_lindblad(rho, H, L, gamma, beta, dt):
    """Apply Lindblad equation with optional SER feedback."""
    commutator = -1j * (H @ rho - rho @ H)
    L_rho_Ldag = L @ rho @ L.conj().T
    Ldag_L = L.conj().T @ L
    lindblad_term = gamma * (L_rho_Ldag - 0.5 * (Ldag_L @ rho + rho @ Ldag_L))

    # SER Feedback
    off_diag = rho - np.diag(np.diag(rho))
    coherence_squared = np.sum(np.abs(off_diag)**2)
    F_rho = np.exp(-coherence_squared)  # Controls feedback strength
    I_minus_rho = I - rho
    feedback_term = (I_minus_rho @ L_rho_Ldag) @ I_minus_rho
    SER_feedback = beta * F_rho * feedback_term

    drho_dt = commutator + lindblad_term + SER_feedback
    return rho + dt * drho_dt

# Store results for different beta values
beta_values = [0, 0.005, 0.01, 0.02, 0.05]  # Sweep of beta values
results = {}

for beta_0 in beta_values:
    # Initialize storage arrays
    purity_vals = np.zeros(num_steps + 1)
    entropy_vals = np.zeros(num_steps + 1)
    coherence_vals = np.zeros(num_steps + 1)
    trace_vals = np.zeros(num_steps + 1)

    # Initialize rho
    rho_m = enforce_positivity(rho_init)
    purity_vals[0], entropy_vals[0], coherence_vals[0], trace_vals[0] = compute_quantities(rho_m)

    for step in range(1, num_steps + 1):
        beta = beta_0 * coherence_vals[step - 1]  # Adaptive beta scaling
        rho_m = ser_lindblad(rho_m, H, L, gamma_value, beta, dt)
        rho_m = enforce_positivity(rho_m)
        purity_vals[step], entropy_vals[step], coherence_vals[step], trace_vals[step] = compute_quantities(rho_m)

    # Store results
    results[beta_0] = {
        "purity": purity_vals,
        "entropy": entropy_vals,
        "coherence": coherence_vals,
        "trace": trace_vals
    }

# Plot Purity Evolution
plt.figure(figsize=(10, 6))
for beta_0 in beta_values:
    plt.plot(time_axis, results[beta_0]["purity"], label=f'Purity (β₀={beta_0})')

plt.xlabel('Time')
plt.ylabel('Purity')
plt.legend()
plt.title('Purity Evolution for Different β₀ Values')
plt.grid(True)
plt.show()

# Plot Coherence Evolution
plt.figure(figsize=(10, 6))
for beta_0 in beta_values:
    plt.plot(time_axis, results[beta_0]["coherence"], label=f'Coherence (β₀={beta_0})')

plt.xlabel('Time')
plt.ylabel('Coherence')
plt.legend()
plt.title('Coherence Evolution for Different β₀ Values')
plt.grid(True)
plt.show()

# Plot Entropy Evolution
plt.figure(figsize=(10, 6))
for beta_0 in beta_values:
    plt.plot(time_axis, results[beta_0]["entropy"], label=f'Entropy (β₀={beta_0})')

plt.xlabel('Time')
plt.ylabel('Entropy')
plt.legend()
plt.title('Entropy Evolution for Different β₀ Values')
plt.grid(True)
plt.show()
