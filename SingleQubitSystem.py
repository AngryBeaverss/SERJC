import numpy as np
import matplotlib.pyplot as plt

##############################################################################
# 1) SIMULATION PARAMETERS
##############################################################################
dt = 0.001          # Time step
total_time = 100.0  # Total simulation time
num_steps = int(total_time / dt)
time_axis = np.linspace(0, total_time, num_steps + 1)
ser_delay = int(num_steps * 0.2)

##############################################################################
# 2) SYSTEM MATRICES (2×2 SINGLE-QUBIT)
##############################################################################
# Pauli matrices and identity
I = np.array([[1, 0], [0, 1]], dtype=complex)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)  # X gate (bit flip)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)  # Energy eigenbasis
sigma_m = np.array([[0, 0], [1, 0]], dtype=complex)  # Lowering operator

# Define Hamiltonian (simple qubit energy splitting)
omega = 1.0  # Qubit frequency
H = (omega / 2) * sigma_z  # Standard qubit Hamiltonian

# Define Lindblad collapse operator for decay
L = sigma_m  # Spontaneous emission operator

# Initial state: Superposition (Work-Retaining)
rho = np.array([[0.7, 0.0], [0.0, 0.3]], dtype=complex)  # Mixed state

##############################################################################
# 3) SIMULATION PARAMETERS FOR SER FEEDBACK
##############################################################################
gamma = 0.1   # Spontaneous emission rate
beta_0 = 0.5  # SER feedback strength (increased)
hbar = 1.0    # Set hbar = 1 for simplicity

##############################################################################
# 4) ARRAYS TO STORE EVOLUTION DATA
##############################################################################
purity_vals = np.zeros(num_steps + 1)
entropy_vals = np.zeros(num_steps + 1)
coherence_vals = np.zeros(num_steps + 1)
energy_vals = np.zeros(num_steps + 1)  # Track energy ⟨H⟩

##############################################################################
# 5) HELPER FUNCTIONS
##############################################################################
def von_neumann_entropy(rho):
    """Compute von Neumann entropy S(ρ) = -Tr(ρ log ρ)."""
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-12]  # Ignore near-zero values
    return -np.sum(eigvals * np.log2(eigvals))

def enforce_positivity(rho):
    """Ensure ρ remains a valid density matrix by clamping negative eigenvalues."""
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.clip(eigvals, 0, None)  # Remove negative eigenvalues
    rho = eigvecs @ np.diag(eigvals) @ eigvecs.conj().T
    return rho / np.trace(rho)  # Normalize trace back to 1

def coherence_measure(rho):
    """Measure coherence as the magnitude of off-diagonal elements."""
    return np.abs(rho[0, 1])

def average_energy(rho, H):
    """Calculate the expectation value ⟨H⟩."""
    return np.real(np.trace(rho @ H))

def lindblad_ser_step(rho):
    """Compute the time derivative dρ/dt using Lindblad evolution + SER feedback."""
    # Standard Lindblad dissipator: LρL† - 1/2 {L†L, ρ}
    lindblad = L @ rho @ L.conj().T - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L)

    # Compute SER feedback function F(ρ)
    pop_diff = np.abs(rho[0, 0] - rho[1, 1])  # Population imbalance
    coherence = coherence_measure(rho)

    # Modified F(ρ) → Drives SER toward work retention
    F_rho = 0.5 * (pop_diff + coherence + 1)

    # Adaptive feedback strength to avoid relaxation
    beta = beta_0 * F_rho * (1 - rho[1,1])

    # SER feedback term: β F(ρ) (I - ρ) LρL† (I - ρ)
    ser_feedback = beta * (I - rho) @ L @ rho @ L.conj().T @ (I - rho)

    # Von Neumann evolution: -i [H, ρ]
    von_neumann = -1j * (H @ rho - rho @ H)

    # Total evolution equation
    d_rho = von_neumann + gamma * lindblad + ser_feedback
    return d_rho

##############################################################################
# 6) MAIN SIMULATION LOOP
##############################################################################
# Store initial values
purity_vals[0] = np.real(np.trace(rho @ rho))  # Tr(ρ²)
entropy_vals[0] = von_neumann_entropy(rho)
coherence_vals[0] = coherence_measure(rho)
energy_vals[0] = average_energy(rho, H)  # Initial Energy

max_energy = energy_vals[0]
max_energy_step = 0  # Store when max energy occurs

# Time evolution
for step in range(num_steps):
    rho += dt * lindblad_ser_step(rho)  # Euler step
    rho = enforce_positivity(rho)  # Ensure positivity

    # Store observables
    purity_vals[step + 1] = np.real(np.trace(rho @ rho))  # Purity Tr(ρ²)
    entropy_vals[step + 1] = von_neumann_entropy(rho)  # Entropy S(ρ)
    coherence_vals[step + 1] = coherence_measure(rho)  # Coherence |ρ01|
    energy_vals[step + 1] = average_energy(rho, H)  # Energy ⟨H⟩

    if step > ser_delay and energy_vals[step + 1] > max_energy:
        # Only start tracking max energy after 10% of simulation
        max_energy = energy_vals[step + 1]
        max_energy_step = step + 1  # Store the step index

##############################################################################
# 7) APPLY UNITARY OPERATION (WORK EXTRACTION)
##############################################################################
# Measure energy before applying unitary
pre_energy = average_energy(rho, H)

# Check if applying the unitary reduces energy
rho_flipped = sigma_x @ rho @ sigma_x  # Apply Pauli-X
post_energy = average_energy(rho_flipped, H)

# Only apply the unitary if it leads to work extraction
if post_energy < pre_energy:
    rho = rho_flipped  # Accept the transition
    work_extracted = pre_energy - post_energy
else:
    work_extracted = 0.0  # No work was extractable

# Store the final energy
final_energy = average_energy(rho, H)

# Compute Efficiency Metrics
E_init = energy_vals[0]  # Initial energy
E_max = max(energy_vals)  # Maximum energy before work extraction
E_min = min(energy_vals)  # Minimum energy reached

eta = work_extracted / E_max if E_max > 0 else 0
eta_recovery = work_extracted / (E_max - E_min) if (E_max - E_min) > 0 else 0
eta_SER = work_extracted / (E_max - E_init) if (E_max - E_init) > 0 else 0

# Print efficiency results
print(f"🔹 Work Extracted: {work_extracted:.6f}")
print(f"🔹 Efficiency η = {eta:.6%}")
print(f"🔹 Energy Recovery Ratio η_recovery = {eta_recovery:.6%}")
print(f"🔹 SER Efficiency η_SER = {eta_SER:.6%}")

##############################################################################
# 8) PLOTTING RESULTS
##############################################################################
plt.figure(figsize=(10, 6))
plt.plot(time_axis, purity_vals, label="Purity (Tr[ρ²])")
plt.plot(time_axis, entropy_vals, label="Entropy (S(ρ))")
plt.plot(time_axis, coherence_vals, label="Coherence |ρ01|")
plt.plot(time_axis, energy_vals, label="Average Energy ⟨H⟩")
plt.axvline(time_axis[max_energy_step], color="red", linestyle="--", label="Work Extraction Time")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.title("SER Feedback with Work Extraction")
plt.grid()
plt.show()

##############################################################################
# 9) PRINT FINAL VALUES
##############################################################################
print(f"Final Purity: {purity_vals[-1]:.6f}")
print(f"Final Entropy: {entropy_vals[-1]:.6f}")
print(f"Final Coherence: {coherence_vals[-1]:.6f}")
print(f"Max Extractable Work: {max_energy:.6f}")
print(f"Final Energy ⟨H⟩ After Work Extraction: {final_energy:.6f}")
print(f"Work Extracted: {work_extracted:.6f}")
