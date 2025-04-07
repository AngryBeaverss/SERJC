import numpy as np
import matplotlib.pyplot as plt


# Load energy rate data
energy_data = np.loadtxt("SER_energy_derivative.csv", delimiter=",", skiprows=1)
time = energy_data[:, 0]  # Time column
energy_rate = energy_data[:, 1]  # dE/dt column

# Compute total energy injected using the trapezoidal rule
total_energy_cost = np.trapezoid(energy_rate, time)

print(f"Total energy injected by SER feedback: {total_energy_cost:.6f}")

cumulative_energy = np.cumsum(energy_rate) * (time[1] - time[0])

plt.figure(figsize=(10, 6))
plt.plot(time, cumulative_energy, label="Total Energy Injected")
plt.xlabel("Time")
plt.ylabel("Injected Energy")
plt.legend()
plt.title("Total Energy Injection Over Time")
plt.grid()
plt.savefig("SER_total_energy.png", dpi=300)
plt.show()
