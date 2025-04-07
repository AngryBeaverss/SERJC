import numpy as np
import matplotlib.pyplot as plt

# Constants
hbar = 1.0
dt = 0.001
total_time = 50
steps = int(total_time / dt)
time = np.linspace(0, total_time, steps)

# Pauli matrices (qubit system)
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
sigma_minus = np.array([[0, 1], [0, 0]], dtype=complex)

# Realistic Hamiltonian (Jaynes-Cummings)
omega_qubit = 1.0
omega_drive = 1.0
drive_strength = 4.0  # previously was 0.1 (too weak)
H = 0.5 * omega_qubit * sigma_z + drive_strength * sigma_x

# Lindblad operator (realistic spontaneous emission)
gamma_spont = 0.01
L = np.sqrt(gamma_spont) * sigma_minus

# Identity operator
I = np.eye(2, dtype=complex)

# Initial state (mixed state)
rho = np.array([[0.6, 0.4], [0.4, 0.4]], dtype=complex)


# Helper functions
def lindblad_ser(rho, H, L, I, gamma, beta, dt):
    commutator = -1j / hbar * (H @ rho - rho @ H)
    lindblad = gamma * (L @ rho @ L.conj().T - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L))

    coherence_squared = np.sum(np.abs(rho) ** 2) - np.sum(np.abs(np.diag(rho)) ** 2)
    F_rho = np.exp(-coherence_squared)
    feedback = beta * F_rho * (I - rho) @ L @ rho @ L.conj().T @ (I - rho)

    drho_dt = commutator + lindblad + feedback
    rho_next = rho + drho_dt * dt

    # Positivity enforcement
    vals, vecs = np.linalg.eigh(rho_next)
    vals_clamped = np.clip(vals, 0, None)
    rho_pos = vecs @ np.diag(vals_clamped) @ vecs.conj().T
    return rho_pos / np.trace(rho_pos)

def coherence_measure(rho):
    return np.abs(rho[0,1]) + np.abs(rho[1,0])


# Storage
purity = np.zeros(steps)
coherence = np.zeros(steps)

# Main simulation loop
for step in range(steps):
    c = coherence_measure(rho)
    gamma = gamma_spont * np.exp(-2 * (1 - c))
    beta = min(0.5 * (1 - np.exp(-c)), 2.0)

    rho = lindblad_ser(rho, H, L, I, gamma, beta, dt)

    purity[step] = np.real(np.trace(rho @ rho))
    coherence[step] = c


# Plot results
plt.figure(figsize=(10, 5))
plt.plot(time, purity, label='Purity')
plt.plot(time, coherence, label='Coherence')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.title('SER Model in Realistic Qubit–Cavity Scenario')
plt.show()
