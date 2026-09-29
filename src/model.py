import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, data):
        for pattern in data:
            self.weights += np.outer(pattern, pattern)
        np.fill_diagonal(self.weights, 0)
        self.weights /= self.size

    def predict(self, pattern, steps=5):
        state = np.copy(pattern)
        for _ in range(steps):
            for i in range(self.size):
                suma = np.dot(self.weights[i], state)
                state[i] = 1 if suma >= 0 else -1
        return state

    def save_weights(self, filepath):
        np.save(filepath, self.weights)

    def load_weights(self, filepath):
        self.weights = np.load(filepath)