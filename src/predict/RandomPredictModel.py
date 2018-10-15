import numpy as np


class RandomPredictModel:
    def __init__(self, k):
        self.k = k

    def fit(self, X, Y):
        pass

    def predict(self, X):
        X_copy = np.delete(X, 4, axis=2)
        for i in range(len(X_copy)):
            matrix = X_copy[i]
            for j in range(self.k):
                offset = self.k - j
                vector_idx = len(matrix) - offset

                vector_prediction_idx = np.random.randint(4)
                vector_prediction = np.zeros(4)
                vector_prediction[vector_prediction_idx] = 1

                matrix[vector_idx] = vector_prediction

        return X_copy
