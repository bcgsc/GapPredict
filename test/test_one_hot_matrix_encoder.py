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

    def test_encode_char_with_quality(self):
        encoder = OneHotMatrixEncoder(1)
        sequences = np.array([
            [1],
            [4],
            [0]
        ])
        quality = np.array([
            [28],
            [29],
            [30]
        ])
        encoded_vectors = encoder.encode_sequences(sequences, quality)
        expected_vectors = np.array([
            [
                [0, 1, 0, 0, 0, 28]
            ],
            [
                [0, 0, 0, 0, 1, 29]
            ],
            [
                [1, 0, 0, 0, 0, 30]
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

    def test_encode_sequences_with_quality(self):
        encoder = OneHotMatrixEncoder(4)
        sequences = np.array([
            [2, 1, 4, 3],
            [1, 4, 3, 2],
            [1, 4, 0, 2]
        ])
        quality = np.array([
            [19, 23, 25, 30],
            [35, 40, 12, 10],
            [35, 40, 13, 10]
        ])
        encoded_vectors = encoder.encode_sequences(sequences, quality)
        expected_vectors = np.array([
            [
                [0, 0, 1, 0, 0, 19],
                [0, 1, 0, 0, 0, 23],
                [0, 0, 0, 0, 1, 25],
                [0, 0, 0, 1, 0, 30]
            ],
            [
                [0, 1, 0, 0, 0, 35],
                [0, 0, 0, 0, 1, 40],
                [0, 0, 0, 1, 0, 12],
                [0, 0, 1, 0, 0, 10]
            ],
            [
                [0, 1, 0, 0, 0, 35],
                [0, 0, 0, 0, 1, 40],
                [1, 0, 0, 0, 0, 13],
                [0, 0, 1, 0, 0, 10]
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

    def test_encode_char_with_quality_and_base_placeholders(self):
        encoder = OneHotMatrixEncoder(1, 3)
        sequences = np.array([
            [1],
            [4],
            [0]
        ])
        quality = np.array([
            [28],
            [29],
            [30]
        ])
        encoded_vectors = encoder.encode_sequences(sequences, quality)
        expected_vectors = np.array([
            [
                [0, 1, 0, 0, 0, 28],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]
            ],
            [
                [0, 0, 0, 0, 1, 29],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]
            ],
            [
                [1, 0, 0, 0, 0, 30],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]
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

    def test_encode_sequences_with_quality_and_base_placeholders(self):
        encoder = OneHotMatrixEncoder(4, 3)
        sequences = np.array([
            [2, 1, 4, 3],
            [1, 4, 3, 2],
            [1, 4, 0, 2]
        ])
        quality = np.array([
            [19, 23, 25, 30],
            [35, 40, 12, 10],
            [35, 40, 13, 10]
        ])
        encoded_vectors = encoder.encode_sequences(sequences, quality)
        expected_vectors = np.array([
            [
                [0, 0, 1, 0, 0, 19],
                [0, 1, 0, 0, 0, 23],
                [0, 0, 0, 0, 1, 25],
                [0, 0, 0, 1, 0, 30],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]
            ],
            [
                [0, 1, 0, 0, 0, 35],
                [0, 0, 0, 0, 1, 40],
                [0, 0, 0, 1, 0, 12],
                [0, 0, 1, 0, 0, 10],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]
            ],
            [
                [0, 1, 0, 0, 0, 35],
                [0, 0, 0, 0, 1, 40],
                [1, 0, 0, 0, 0, 13],
                [0, 0, 1, 0, 0, 10],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0]
            ]
        ])
        np.testing.assert_array_equal(encoded_vectors, expected_vectors)

