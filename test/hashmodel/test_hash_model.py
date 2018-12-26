import numpy as np

from unittest import TestCase
from predict.heuristic.HashModel import HashModel

class TestHashModel(TestCase):
    def test_predict(self):
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
        model = HashModel()
        model.fit(X, y)

        X_test = [
            "A",
            "T",
            "G",
            "C",
            "A"
        ]

        expected_y = [
            "T",
            "G",
            "G",
            "",
            "T"
        ]
        y = model.predict(X_test)

        np.testing.assert_array_equal(expected_y, y)