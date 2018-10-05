import numpy as np
from unittest import TestCase

from OneHotVectorEncoder import OneHotVectorEncoder, NonpositiveLengthException


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
        sequences = [
            ["A"],
            ["T"]
        ]
        encoded_vectors = encoder.encode_sequences(sequences)
        expected_vectors = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ])
        np.testing.assert_array_equal(encoded_vectors, expected_vectors)

    def test_decode_char(self):
        encoder = OneHotVectorEncoder(1)
        encoded_sequences = np.array([
            [1, 0, 0, 0],
            [0, 0, 0, 1]
        ])
        decoded_sequences = encoder.decode_sequences(encoded_sequences)
        expected_sequences = [
            ["A"],
            ["T"]
        ]
        np.testing.assert_array_equal(decoded_sequences, expected_sequences)

    def test_encode_sequences(self):
        encoder = OneHotVectorEncoder(4)
        sequences = [
            ["C", "A", "T", "G"],
            ["A", "T", "G", "C"]
        ]
        encoded_vectors = encoder.encode_sequences(sequences)
        expected_vectors = np.array([
            [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
        ])
        np.testing.assert_array_equal(encoded_vectors, expected_vectors)

    def test_decode_sequences(self):
        encoder = OneHotVectorEncoder(4)
        encoded_sequences = np.array([
            [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0]
        ])
        decoded_sequences = encoder.decode_sequences(encoded_sequences)
        expected_sequences = [
            ["C", "A", "T", "G"],
            ["A", "T", "G", "C"]
        ]
        np.testing.assert_array_equal(decoded_sequences, expected_sequences)
