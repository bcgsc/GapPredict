import numpy as np

import constants.EncodingConstants as CONSTANTS


class RandomPredictModel:
    def __init__(self, k):
        self.k = k

    def fit(self, X, Y):
        pass

    def predict(self, X):
        num_bases = len(CONSTANTS.BASES)
        dimensions = (len(X), self.k, num_bases+1)
        predictions = np.zeros(dimensions)
        for i in range(len(X)):
            base_idx = np.random.randint(low=1, high=num_bases+1, size=self.k)
            for j in range(self.k):
                predictions[i][j][base_idx[j]] = 1

        return predictions
