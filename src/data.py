import numpy as np

T_PATTERN = np.array([1, 1, 1, -1, 1, -1, -1, 1, -1])
L_PATTERN = np.array([1, -1, -1, 1, -1, -1, 1, 1, 1])

T_NOISY = np.array([1, -1, 1, -1, 1, -1, -1, 1, -1])

def get_training_data():
    return [T_PATTERN, L_PATTERN]

def get_test_data():
    return {"noisy": T_NOISY, "original": T_PATTERN}