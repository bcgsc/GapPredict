import numpy as np

import constants.EncodingConstants as CONSTANTS


class RandomPredictModel:
    def __init__(self, k, has_quality=True):
        self.k = k
        self.has_quality = has_quality

    def fit(self, X, Y):
        pass

    def predict(self, X):
        num_bases = len(CONSTANTS.BASES)
        dimensions = (len(X), self.k, num_bases)
        predictions = np.zeros(dimensions)
        for i in range(len(X)):
            base_idx = np.random.randint(low=0, high=num_bases, size=self.k)
            for j in range(self.k):
                predictions[i][j][base_idx[j]] = 1

        return predictions
