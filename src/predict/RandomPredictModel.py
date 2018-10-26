import numpy as np

from onehot.OneHotMatrix import BASE_ENCODING_IDX_MAP


class RandomPredictModel:
    def __init__(self, k, has_quality=True):
        self.k = k
        self.has_quality = has_quality


    def fit(self, X, Y):
        pass

    def predict(self, X):
        if self.has_quality:
            X_copy = np.delete(X, len(BASE_ENCODING_IDX_MAP), axis=2)
        for i in range(len(X_copy)):
            matrix = X_copy[i]
            for j in range(self.k):
                offset = self.k - j
                vector_idx = len(matrix) - offset

                vector_prediction_idx = np.random.randint(4)
                vector_prediction = np.zeros(len(BASE_ENCODING_IDX_MAP))
                vector_prediction[vector_prediction_idx+1] = 1

                matrix[vector_idx] = vector_prediction

        return X_copy
