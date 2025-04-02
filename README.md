# Quantum Work Extraction with Structured Energy Return (SER)

This project explores **extractable work (ergotropy)** in open quantum systems using a feedback control model called **Structured Energy Return (SER)**. It extends quantum thermodynamics simulations using the Jaynes-Cummings model with realistic decoherence and feedback delay.

## 🔬 Key Idea

> Entanglement alone doesn't preserve energy.  
> SER feedback continuously reshapes coherence after a real-world delay, stabilizing usable quantum energy (ergotropy) over time.

## 📈 What This Project Shows

- How quantum systems lose extractable energy under decoherence
- How delayed feedback based on **concurrence** can counteract that loss
- That **concurrence and ergotropy are decoupled** — proving SER’s unique role

## 🧠 Why It Matters

Quantum batteries, quantum heat engines, and coherent control protocols all face one shared threat: **decoherence kills extractable work**.

SER demonstrates a **practical strategy for energy retention**, even with noisy systems and delayed control — aligning with real hardware limits.

---

## 🛠️ Features

- Full Lindblad master equation with Jaynes-Cummings interaction
- Feedback delay (\( \tau_f \)) to simulate classical signal lag
- Continuous feedback strength modulation: \( \beta(t) = F(\text{concurrence}) \)
- Calculation of:
  - Concurrence (entanglement)
  - Ergotropy (extractable work)
  - Photon number
  - Qubit excitation
  - Correlation plots (concurrence vs. work)

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/AngryBeaverss/SERJC
cd SERJC

# (Optional) create a virtual environment
python -m venv venv && source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run main simulation
python ha.py
```

## Running parameter sweeps

If you want to simulate more than a single value for beta or coupling strength, simply modify

```bash
Feedback levels to test
feedback_strengths = [0.0]  # Different β_max values
```
```bash
Coupling strengths to test
coupling_strengths = [20 / GHz_to_MHz]  # Weak, strong, ultra-strong
```

The code is already set up for multiple runs at a time.
