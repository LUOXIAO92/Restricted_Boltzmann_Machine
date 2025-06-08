import numpy as np

def sigmoid(x : np.ndarray):
    return (1 / (1 + np.exp(-x))).astype(x.dtype)

def binormal_distribution(p : np.ndarray):
    rand  = np.random.random(size = p.shape, dtype = p.dtype)
    state = np.zeros_like(p)
    state[rand < p] = 1.0
    del rand
    return state