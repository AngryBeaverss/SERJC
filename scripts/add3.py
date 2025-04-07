import numpy as np
import subprocess
from scipy.optimize import curve_fit
import sys
import matplotlib as plt

beta_values = np.linspace(0.1, 2.0, 10)  # Test 10 values

# Store results
lambda_values = []
energy_values = []

for beta in beta_values:
    print(f"Running simulation for β = {beta:.2f}")

    # Run the simulation with the new beta value
    subprocess.run([sys.executable, "SERScaling.py", str(beta)])

    # Load entropy decay data
    entropy_data = np.loadtxt("SER_entropy_derivative.csv", delimiter=",", skiprows=1)
    time = entropy_data[:, 0]
    entropy_vals = np.cumsum(entropy_data[:, 1]) * (time[1] - time[0])

    # Fit entropy decay
    def entropy_decay_model(t, S0, lambda_, S_inf):
        return S0 * np.exp(-lambda_ * t) + S_inf


    popt, _ = curve_fit(entropy_decay_model, time, entropy_vals, p0=[1.0, 0.1, 0.0])
    lambda_values.append(popt[1])  # Store decay rate λ

    # Load energy injection data
    energy_data = np.loadtxt("SER_energy_derivative.csv", delimiter=",", skiprows=1)
    total_energy = np.trapezoid(energy_data[:, 1], energy_data[:, 0])
    energy_values.append(total_energy)

    print(f"β = {beta:.2f}, λ = {popt[1]:.4f}, Total Energy = {total_energy:.6f}")

# Save results
np.savetxt("SER_beta_vs_lambda.csv", np.column_stack((beta_values, lambda_values, energy_values)),
           delimiter=",", header="Beta,Entropy_Decay_Rate,Total_Energy", comments="")


