from unittest import TestCase

import numpy as np

from exceptions.NegativePredictionLengthException import NegativePredictionLengthException
from exceptions.NonpositiveLengthException import NonpositiveLengthException
from onehot.OneHotMatrix import OneHotMatrixEncoder


class TestOneHotMatrixEncoder(TestCase):
    def test_encode_bad_length(self):
        try:
            encoder = OneHotMatrixEncoder(0)
        except NonpositiveLengthException as e:
            pass
        except Exception as e:
            self.fail()

    def test_encode_bad_prediction_length(self):
        try:
            encoder = OneHotMatrixEncoder(1, -1)
        except NegativePredictionLengthException as e:
            pass
        except Exception as e:
            self.fail()

    def test_encode_char(self):
        encoder = OneHotMatrixEncoder(1)
        sequences = np.array([
            [1],
            [4],
            [0]
        ])
        encoded_vectors = encoder.encode_sequences(sequences)
        expected_vectors = np.array([
            [
                [0, 1, 0, 0, 0]
            ],
            [
                [0, 0, 0, 0, 1]
            ],
            [
                [1, 0, 0, 0, 0]
            ]
        ])
        np.testing.assert_array_equal(encoded_vectors, expected_vectors)

    def test_encode_sequences(self):
        encoder = OneHotMatrixEncoder(4)
        sequences = np.array([
            [2, 1, 4, 3],
            [1, 4, 3, 2],
            [1, 4, 0, 2]
        ])
        encoded_vectors = encoder.encode_sequences(sequences)
        expected_vectors = np.array([
            [
                [0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0]
            ],
            [
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0]
            ],
            [
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0]
            ]
        ])
        np.testing.assert_array_equal(encoded_vectors, expected_vectors)

    def test_encode_char_and_base_placeholders(self):
        encoder = OneHotMatrixEncoder(1, 3)
        sequences = np.array([
            [1],
            [4],
            [0]
        ])
        encoded_vectors = encoder.encode_sequences(sequences)
        expected_vectors = np.array([
            [
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ],
            [
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ],
            [
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ]
        ])
        np.testing.assert_array_equal(encoded_vectors, expected_vectors)

    def test_encode_sequences_and_base_placeholders(self):
        encoder = OneHotMatrixEncoder(4, 3)
        sequences = np.array([
            [2, 1, 4, 3],
            [1, 4, 3, 2],
            [1, 4, 0, 2]
        ])
        encoded_vectors = encoder.encode_sequences(sequences)
        expected_vectors = np.array([
            [
                [0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ],
            [
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ],
            [
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ]
        ])
        np.testing.assert_array_equal(encoded_vectors, expected_vectors)

    def test_encode_empty_array(self):
        encoder = OneHotMatrixEncoder(4)
        sequences = np.array([])
        encoded_vectors = encoder.encode_sequences(sequences)
        expected_vectors = np.array([])
        np.testing.assert_array_equal(encoded_vectors, expected_vectors)
