import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import DensityMatrix, partial_trace
from scipy.linalg import eigvals

# Simulator setup
sim = AerSimulator(method='statevector')
omega = 1.0
H = 0.5 * omega * np.array([[1, 0], [0, -1]])
threshold_concurrence = 0.9
steps = 20
log = []

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

psi = None
for step in range(steps):
    qc = QuantumCircuit(2)
    if step == 0:
        qc.h(0)
        qc.cx(0, 1)
    else:
        qc.set_statevector(psi)

    angle = np.random.normal(loc=0, scale=np.pi / 16)
    qc.ry(angle, 1)
    qc.save_statevector(label='degraded')
    qc.cx(0, 1)
    qc.save_statevector(label='after_feedback')

    compiled = transpile(qc, sim)
    result = sim.run(compiled).result()

    psi_degraded = result.data(0)['degraded']
    psi_feedback = result.data(0)['after_feedback']
    psi = psi_feedback

    rho_d = DensityMatrix(psi_degraded)
    rho_f = DensityMatrix(psi_feedback)

    rho0_d = partial_trace(rho_d, [1])
    rho0_f = partial_trace(rho_f, [1])

    C_d = concurrence(rho_d.data)
    C_f = concurrence(rho_f.data)
    E_d = ergotropy(rho0_d.data, H)
    E_f = ergotropy(rho0_f.data, H)

    log.append({
        "step": step,
        "concurrence_before": C_d,
        "concurrence_after": C_f,
        "ergotropy_before": E_d,
        "ergotropy_after": E_f,
        "feedback_triggered": int(C_d < threshold_concurrence)
    })

df = pd.DataFrame(log)

# Plot
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axs[0].plot(df['step'], df['concurrence_before'], label='Concurrence Before')
axs[0].plot(df['step'], df['concurrence_after'], '--', label='Concurrence After')
axs[0].legend()
axs[0].grid()

axs[1].plot(df['step'], df['ergotropy_before'], label='Ergotropy Before')
axs[1].plot(df['step'], df['ergotropy_after'], '--', label='Ergotropy After')
axs[1].legend()
axs[1].grid()

plt.xlabel("Step")
plt.tight_layout()
plt.show()
