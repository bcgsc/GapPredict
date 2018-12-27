import numpy as np

from unittest import TestCase
from predict.heuristic.HashModel import HashModel
import constants.EncodingConstants as CONSTANTS

class TestHashModel(TestCase):
    def test_predict(self):
        np.random.seed(0)
        bases_to_predict = 1
        X = [
            "A",
            "A",
            "T",
            "T",
            "G",
            "A",
            "G",
            "T",
            "A"
        ]
        y = [
            "T",
            "T",
            "G",
            "A",
            "G",
            "T",
            "C",
            "G",
            "C"
        ]
        model = HashModel(bases_to_predict)
        model.fit(X, y)

        X_test = [
            "A",
            "T",
            "G",
            "C",
            "A"
        ]

        y = model.predict(X_test)

        np.random.seed(0)
        prediction_idx = np.random.randint(4, size=bases_to_predict)
        random_prediction = "".join(CONSTANTS.REVERSE_INTEGER_ENCODING[prediction_idx])

        expected_y = [
            "T",
            "G",
            "G",
            random_prediction,
            "T"
        ]

        np.testing.assert_array_equal(expected_y, y)