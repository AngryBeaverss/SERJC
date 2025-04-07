
import numpy as np
from matplotlib import pyplot as plt

##############################################################################
# 1) ADAPTIVE TIMESTEP SIMULATION PARAMETERS
##############################################################################
total_time = 100.0
initial_dt = 0.001
max_dt = 0.005
min_dt = 0.0001
time_axis = [0]

##############################################################################
# 2) SYSTEM SIZE SELECTION
##############################################################################
DSIZE = int(input("Enter matrix size (2 for 2x2, 4 for 4x4): ").strip())

##############################################################################
# 3) PHYSICALLY MOTIVATED OPERATORS (Example for 2x2 Qubit)
##############################################################################
if DSIZE == 2:
    # Example Hamiltonian: Qubit in magnetic field (Pauli-X)
    H_init = np.array([[0, 1], [1, 0]], dtype=np.complex128)

    # Example Lindblad operator: decay operator (Pauli lowering operator)
    L_init = np.array([[0, 1], [0, 0]], dtype=np.complex128)
else:
    # Random operators for DSIZE > 2
    H_init = np.random.rand(DSIZE, DSIZE) + 1j * np.random.rand(DSIZE, DSIZE)
    H_init = (H_init + H_init.conj().T) / 2

    L_init = np.random.rand(DSIZE, DSIZE) + 1j * np.random.rand(DSIZE, DSIZE)
    L_init = (L_init + L_init.conj().T) / 2

I_init = np.eye(DSIZE, dtype=np.complex128)

A = np.random.randn(DSIZE, DSIZE) + 1j * np.random.randn(DSIZE, DSIZE)
rho_init = A @ A.conj().T
rho_init /= np.trace(rho_init)

##############################################################################
# 4) HELPER FUNCTIONS (Unchanged)
##############################################################################
def compute_quantities(rho_m):
    rho2 = rho_m @ rho_m
    purity = np.real(np.trace(rho2))
    w = np.linalg.eigvalsh(rho_m)
    w_pos = w[w > 1e-12]
    entropy = -np.sum(w_pos * np.log2(w_pos)) if len(w_pos) > 0 else 0.0
    coherence = np.sum(np.abs(rho_m)) - np.trace(np.abs(rho_m))
    trace = np.real(np.trace(rho_m))
    return purity, entropy, coherence, trace

def enforce_positivity(rho_m):
    w, v = np.linalg.eigh(rho_m)
    w_clamped = np.clip(w, 0, None)
    rho_corrected = v @ np.diag(w_clamped) @ v.conj().T
    return rho_corrected / np.trace(rho_corrected)

def ser_lindblad(rho, H, L, I, hbar, gamma, beta, dt):
    commutator = -1j / hbar * (H @ rho - rho @ H)
    lindblad_term = gamma * (L @ rho @ L.conj().T - 0.5 * (L.conj().T @ L @ rho + rho @ L.conj().T @ L))
    coherence_squared = np.sum(np.abs(rho)**2) - np.sum(np.abs(np.diag(rho))**2)
    F_rho = np.exp(-coherence_squared)
    temp = (I - rho) @ (L @ rho @ L.conj().T)
    SER_feedback = beta * F_rho * temp @ (I - rho)
    drho_dt = commutator + lindblad_term + SER_feedback
    return rho + dt * drho_dt

##############################################################################
# 5) INITIALIZATION
##############################################################################
rho_m = enforce_positivity(rho_init.copy())
purity_vals, entropy_vals, coherence_vals, trace_vals = [], [], [], []
hbar, current_time, dt = 1.0, 0.0, initial_dt

##############################################################################
# 6) ADAPTIVE TIME-STEPPING MAIN LOOP
##############################################################################
while current_time < total_time:
    p, e, c, t = compute_quantities(rho_m)
    purity_vals.append(p)
    entropy_vals.append(e)
    coherence_vals.append(np.real(c))
    trace_vals.append(t)
    time_axis.append(current_time)

    gamma = 0.3 * np.exp(-2.0 * (1.0 - c))
    beta = min(0.5 * (1 - np.exp(-c)), 2.0)

    rho_next = ser_lindblad(rho_m, H_init, L_init, I_init, hbar, gamma, beta, dt)
    rho_next = enforce_positivity(rho_next)

    coherence_change = abs(c - compute_quantities(rho_next)[2])
    dt = max(min_dt, min(max_dt, initial_dt / (1 + 10 * coherence_change)))

    current_time += dt
    rho_m = rho_next

    if len(time_axis) % 5000 == 0:
        print(f"Time {current_time:.2f}, Purity={p:.4f}, Entropy={e:.4f}, Coherence={c:.4f}, Trace={t:.4f}, dt={dt:.5f}")

##############################################################################
# 7) FINAL OUTPUT AND PLOTTING
##############################################################################
plt.figure(figsize=(10, 6))
plt.plot(time_axis[:-1], purity_vals, label='Purity')
plt.plot(time_axis[:-1], entropy_vals, label='Entropy')
plt.plot(time_axis[:-1], coherence_vals, label='Coherence')
plt.xlabel('Time')
plt.ylabel('Quantity')
plt.legend()
plt.grid(True)
plt.title('Adaptive Evolution of Purity, Entropy, Coherence')
plt.show()
