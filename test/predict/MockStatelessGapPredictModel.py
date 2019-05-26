import numpy as np

class MockStatelessGapPredictModel:
    def __init__(self):
        pass

    def reset_states(self):
        pass

    def predict(self, X):
        samples = len(X)
        out = np.zeros((samples, 4))
        for i in range(len(X)):
            out[i] = self.map_vector(X[i])
        return out

    def map_vector(self, x):
        if np.array_equal(x, np.array([0, 2, 3, 1, 1, 2, 0])):
            return np.array([np.exp(-0.3), np.exp(-0.4), np.exp(-1), np.exp(-4)])
        elif np.array_equal(x, np.array([0, 2, 3, 1, 1, 2, 0, 0])):
            return np.array([np.exp(-1.3), np.exp(-1.4), np.exp(-1.5), np.exp(-1.6)])
        elif np.array_equal(x, np.array([0, 2, 3, 1, 1, 2, 0, 1])):
            return np.array([np.exp(-0.7), np.exp(-0.8), np.exp(-0.9), np.exp(-1)])
        elif np.array_equal(x, np.array([0, 2, 3, 1, 1, 2, 0, 2])):
            return np.array([np.exp(-2), np.exp(0), np.exp(-3), np.exp(-3)])
        elif np.array_equal(x, np.array([0, 2, 3, 1, 1, 2, 0, 3])):
            return np.array([np.exp(-1), np.exp(-0.5), np.exp(-0.8), np.exp(-0.2)])