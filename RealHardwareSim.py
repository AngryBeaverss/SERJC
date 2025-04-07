import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix, partial_trace
from qiskit_aer.noise import NoiseModel, thermal_relaxation_error
from qiskit_aer.noise.errors import depolarizing_error, ReadoutError
from scipy.linalg import eigvals

# --- Configuration ---
steps = 20
feedback_delay_step = 1  # τ_f delay
omega = 1.0
H = 0.5 * omega * np.array([[1, 0], [0, -1]])
threshold_concurrence = 0.999999

# --- Noise Model ---
noise_model = NoiseModel()
T1, T2, gate_time = 50_000, 70_000, 100
thermal_error = thermal_relaxation_error(T1, T2, gate_time)
noise_model.add_all_qubit_quantum_error(thermal_error, ['ry', 'rx'])
cx_error = depolarizing_error(0.01, 2)
noise_model.add_all_qubit_quantum_error(cx_error, ['cx'])
readout_matrix = [[0.98, 0.02], [0.02, 0.98]]
readout_error = ReadoutError(readout_matrix)
noise_model.add_readout_error(readout_error, [0])
noise_model.add_readout_error(readout_error, [1])

sim = AerSimulator(noise_model=noise_model)
log = []

# --- Helper Functions ---
def concurrence(rho_full):
    Y = np.array([[0, -1j], [1j, 0]])
    YY = np.kron(Y, Y)
    R = rho_full @ YY @ np.conj(rho_full) @ YY
    eigs = np.sort(np.sqrt(np.maximum(0, np.real(eigvals(R)))))[::-1]
    return max(0, eigs[0] - sum(eigs[1:]))

def ergotropy(rho_red, H):
    p = np.sort(np.real(np.linalg.eigvalsh(rho_red)))[::-1]
    e = np.sort(np.real(np.linalg.eigvalsh(H)))
    energy_initial = np.sum(p * np.diag(H))
    energy_passive = np.sum(p * e)
    return energy_initial - energy_passive

def F_ser(C):
    return (1 - C) * np.exp(-C)

# --- Main Simulation Loop ---
psi = None
for step in range(steps):
    qc = QuantumCircuit(2)

    if step == 0:
        qc.h(0)
        qc.cx(0, 1)
        qc.rx(np.pi / 8, 1)
    else:
        qc.set_statevector(psi)

    # Apply degradation (simulate loss)
    angle = np.random.normal(loc=0, scale=np.pi / 4)
    qc.ry(angle, 1)
    qc.rx(np.pi / 3, 1)
    qc.save_statevector(label='degraded')

    # Simulate and calculate degraded state metrics
    compiled = transpile(qc, sim)
    result = sim.run(compiled).result()
    psi_degraded = result.data(0)['degraded']
    rho_degraded = DensityMatrix(psi_degraded)
    C_d = concurrence(rho_degraded.data)
    F_raw = F_ser(C_d)
    theta_f = max(min(F_raw * np.pi * 10, np.pi / 2), 0.05)
    print(
        f"[Step {step}] C_d: {C_d:.4f}, theta_f: {theta_f:.4f}, Feedback: {step >= feedback_delay_step and C_d < threshold_concurrence}")

    # Apply adaptive feedback if delay condition is met
    if step >= feedback_delay_step and C_d < threshold_concurrence:
        qc.rx(theta_f, 1)

    qc.save_statevector(label='after_feedback')
    compiled = transpile(qc, sim)
    result = sim.run(compiled).result()
    psi_feedback = result.data(0)['after_feedback']
    psi = psi_feedback

    rho_feedback = DensityMatrix(psi_feedback)
    rho0_d = partial_trace(rho_degraded, [1])
    rho0_f = partial_trace(rho_feedback, [1])
    C_f = concurrence(rho_feedback.data)
    E_d = ergotropy(rho0_d.data, H)
    E_f = ergotropy(rho0_f.data, H)

    log.append({
        "step": step,
        "concurrence_before": C_d,
        "concurrence_after": C_f,
        "ergotropy_before": E_d,
        "ergotropy_after": E_f,
        "theta_feedback": theta_f,
        "feedback_triggered": int(step >= feedback_delay_step and C_d < threshold_concurrence)
    })

# --- Visualization ---
df = pd.DataFrame(log)

fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axs[0].plot(df['step'], df['concurrence_before'], label='Concurrence Before')
axs[0].plot(df['step'], df['concurrence_after'], '--', label='Concurrence After')
axs[0].legend(); axs[0].grid(); axs[0].set_ylabel("Concurrence")

axs[1].plot(df['step'], df['ergotropy_before'], label='Ergotropy Before')
axs[1].plot(df['step'], df['ergotropy_after'], '--', label='Ergotropy After')
axs[1].legend(); axs[1].grid(); axs[1].set_ylabel("Ergotropy")
axs[1].set_xlabel("Step")

plt.suptitle("Polished SER Simulation with Feedback Delay and F(C) Adaptive Recovery")
plt.tight_layout()
plt.show()
