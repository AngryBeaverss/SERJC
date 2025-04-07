import numpy as np

def adaptive_feedback(beta_max, entanglement):
    return min(beta_max * (1 - entanglement) * np.exp(-entanglement), 0.02)
