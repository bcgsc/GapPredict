from unittest import TestCase

import numpy as np

from exceptions.NonpositiveLengthException import NonpositiveLengthException
from onehot.OneHotVector import OneHotVectorEncoder


class TestOneHotVectorEncoder(TestCase):
    def test_encode_bad_length(self):
        try:
            encoder = OneHotVectorEncoder(0)
        except NonpositiveLengthException as e:
            pass
        except Exception as e:
            self.fail()

    def test_encode_char(self):
        encoder = OneHotVectorEncoder(1)
        sequences = np.array([
            [0],
            [3]
        ])
        encoded_vectors = encoder.encode_sequences(sequences)
        expected_vectors = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ])
        np.testing.assert_array_equal(encoded_vectors, expected_vectors)

    def test_encode_sequences(self):
        encoder = OneHotVectorEncoder(4)
        sequences = np.array([
            [1, 0, 3, 2],
            [0, 3, 2, 1]
        ])
        encoded_vectors = encoder.encode_sequences(sequences)
        expected_vectors = np.array([
            [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
        ])
        np.testing.assert_array_equal(encoded_vectors, expected_vectors)

    def test_encode_empty_array(self):
        encoder = OneHotVectorEncoder(4)
        sequences = np.array([])
        encoded_vectors = encoder.encode_sequences(sequences)
        expected_vectors = np.array([])
        np.testing.assert_array_equal(encoded_vectors, expected_vectors)
