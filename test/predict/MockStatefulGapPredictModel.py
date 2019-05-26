import numpy as np

class MockStatefulGapPredictModel:
    def __init__(self):
        self.state = []

    def reset_states(self):
        self.state = []

    def predict(self, X):
        samples = len(X)
        out = np.zeros((samples, 4))
        if len(self.state) == 0:
            for i in range(samples):
                self.state.append(list(X[i]))
        else:
            for i in range(samples):
                self.state[i] += list(X[i])
        for i in range(samples):
            out[i] = self.map_vector(self.state[i])
        return out

    def map_vector(self, x):
        np_x = np.array(x)
        if np.array_equal(np_x, np.array([0, 2, 3, 1, 1, 2, 0])):
            return np.array([np.exp(-0.3), np.exp(-0.4), np.exp(-1), np.exp(-4)])
        elif np.array_equal(np_x, np.array([0, 2, 3, 1, 1, 2, 0, 0])):
            return np.array([np.exp(-1.3), np.exp(-1.4), np.exp(-1.5), np.exp(-1.6)])
        elif np.array_equal(np_x, np.array([0, 2, 3, 1, 1, 2, 0, 1])):
            return np.array([np.exp(-0.8), np.exp(-0.7), np.exp(-0.9), np.exp(-1)])
        elif np.array_equal(np_x, np.array([0, 2, 3, 1, 1, 2, 0, 2])):
            return np.array([np.exp(-2), np.exp(0), np.exp(-3), np.exp(-3)])
        elif np.array_equal(np_x, np.array([0, 2, 3, 1, 1, 2, 0, 3])):
            return np.array([np.exp(-1), np.exp(-0.5), np.exp(-0.8), np.exp(-0.2)])