import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Define exponential decay function
def entropy_decay_model(t, S0, lambda_, S_inf):
    return S0 * np.exp(-lambda_ * t) + S_inf


# Load entropy data
data = np.loadtxt("SER_entropy_derivative.csv", delimiter=",", skiprows=1)
time = data[:, 0]  # Time column
entropy_vals = np.cumsum(data[:, 1]) * (time[1] - time[0])  # Integrate dS/dt to get S(t)

# Normalize entropy (ensuring it starts from max value)
entropy_vals -= np.min(entropy_vals)
entropy_vals /= np.max(entropy_vals)

# Perform curve fitting
popt, _ = curve_fit(entropy_decay_model, time, entropy_vals, p0=[1.0, 0.1, 0.0], maxfev=5000)


# Extract fit parameters
S0, lambda_, S_inf = popt
print(f"Fitted parameters: S0 = {S0:.4f}, lambda = {lambda_:.4f}, S_inf = {S_inf:.4f}")

# Generate fitted curve
entropy_fit = entropy_decay_model(time, *popt)

plt.figure(figsize=(10, 6))
plt.plot(time, entropy_vals, 'o', markersize=3, label='Simulated Entropy')
plt.plot(time, entropy_fit, 'r-', label=f'Fit: S0={S0:.2f}, λ={lambda_:.2f}')
plt.xlabel("Time")
plt.ylabel("Entropy (S)")
plt.legend()
plt.title("Entropy Decay Fit")
plt.grid()
plt.savefig("SER_entropy_fit.png", dpi=300)
plt.show()

# Load results
data = np.loadtxt("SER_beta_vs_lambda.csv", delimiter=",", skiprows=1)
beta_values, lambda_values, energy_values = data[:, 0], data[:, 1], data[:, 2]

# Plot entropy decay rate vs. feedback strength
plt.figure(figsize=(10, 6))
plt.plot(beta_values, lambda_values, 'o-', label="Entropy Decay Rate (λ)")
plt.xlabel("Feedback Strength (β)")
plt.ylabel("Decay Rate (λ)")
plt.title("Effect of SER Feedback on Entropy Decay")
plt.grid(True)
plt.legend()
plt.savefig("SER_beta_vs_lambda.png", dpi=300)
plt.show()

# Plot total energy vs. beta
plt.figure(figsize=(10, 6))
plt.plot(beta_values, energy_values, 'o-', label="Total Energy Injected")
plt.xlabel("Feedback Strength (β)")
plt.ylabel("Total Energy Injected")
plt.title("Energy Cost vs. Feedback Strength")
plt.grid(True)
plt.legend()
plt.savefig("SER_beta_vs_energy.png", dpi=300)
plt.show()