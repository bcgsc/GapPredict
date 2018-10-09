import numpy as np
from unittest import TestCase

from exceptions.NonpositiveLengthException import NonpositiveLengthException
from onehot.OneHotMatrixEncoder import OneHotMatrixEncoder


class TestOneHotMatrixEncoder(TestCase):
    def test_encode_bad_length(self):
        try:
            encoder = OneHotMatrixEncoder(0)
        except NonpositiveLengthException as e:
            pass
        except Exception as e:
            self.fail()

    def test_encode_char(self):
        encoder = OneHotMatrixEncoder(1)
        sequences = [
            ["A"],
            ["T"]
        ]
        encoded_vectors = encoder.encode_sequences(sequences)
        expected_vectors = np.array([
            [
                [1, 0, 0, 0]
            ],
            [
                [0, 0, 0, 1]
            ]
        ])
        np.testing.assert_array_equal(encoded_vectors, expected_vectors)

    def test_encode_char_with_quality(self):
        encoder = OneHotMatrixEncoder(1)
        sequences = [
            ["A"],
            ["T"]
        ]
        quality = np.array([
            [28],
            [29]
        ])
        encoded_vectors = encoder.encode_sequences(sequences, quality)
        expected_vectors = np.array([
            [
                [1, 0, 0, 0, 28]
            ],
            [
                [0, 0, 0, 1, 29]
            ]
        ])
        np.testing.assert_array_equal(encoded_vectors, expected_vectors)

    def test_encode_sequences(self):
        encoder = OneHotMatrixEncoder(4)
        sequences = [
            ["C", "A", "T", "G"],
            ["A", "T", "G", "C"]
        ]
        encoded_vectors = encoder.encode_sequences(sequences)
        expected_vectors = np.array([
            [
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]
            ],
            [
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0]
            ]
        ])
        np.testing.assert_array_equal(encoded_vectors, expected_vectors)

    def test_encode_sequences_with_quality(self):
        encoder = OneHotMatrixEncoder(4)
        sequences = [
            ["C", "A", "T", "G"],
            ["A", "T", "G", "C"]
        ]
        quality = np.array([
            [19, 23, 25, 30],
            [35, 40, 12, 10]
        ])
        encoded_vectors = encoder.encode_sequences(sequences, quality)
        expected_vectors = np.array([
            [
                [0, 1, 0, 0, 19],
                [1, 0, 0, 0, 23],
                [0, 0, 0, 1, 25],
                [0, 0, 1, 0, 30]
            ],
            [
                [1, 0, 0, 0, 35],
                [0, 0, 0, 1, 40],
                [0, 0, 1, 0, 12],
                [0, 1, 0, 0, 10]
            ]
        ])
        np.testing.assert_array_equal(encoded_vectors, expected_vectors)

    def test_encode_char_and_base_placeholders(self):
        encoder = OneHotMatrixEncoder(1, 3)
        sequences = [
            ["A"],
            ["T"]
        ]
        encoded_vectors = encoder.encode_sequences(sequences)
        expected_vectors = np.array([
            [
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ],
            [
                [0, 0, 0, 1],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ]
        ])
        np.testing.assert_array_equal(encoded_vectors, expected_vectors)

    def test_encode_char_with_quality_and_base_placeholders(self):
        encoder = OneHotMatrixEncoder(1, 3)
        sequences = [
            ["A"],
            ["T"]
        ]
        quality = np.array([
            [28],
            [29]
        ])
        encoded_vectors = encoder.encode_sequences(sequences, quality)
        expected_vectors = np.array([
            [
                [1, 0, 0, 0, 28],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ],
            [
                [0, 0, 0, 1, 29],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ]
        ])
        np.testing.assert_array_equal(encoded_vectors, expected_vectors)

    def test_encode_sequences_and_base_placeholders(self):
        encoder = OneHotMatrixEncoder(4, 3)
        sequences = [
            ["C", "A", "T", "G"],
            ["A", "T", "G", "C"]
        ]
        encoded_vectors = encoder.encode_sequences(sequences)
        expected_vectors = np.array([
            [
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ],
            [
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0]
            ]
        ])
        np.testing.assert_array_equal(encoded_vectors, expected_vectors)

    def test_encode_sequences_with_quality_and_base_placeholders(self):
        encoder = OneHotMatrixEncoder(4, 3)
        sequences = [
            ["C", "A", "T", "G"],
            ["A", "T", "G", "C"]
        ]
        quality = np.array([
            [19, 23, 25, 30],
            [35, 40, 12, 10]
        ])
        encoded_vectors = encoder.encode_sequences(sequences, quality)
        expected_vectors = np.array([
            [
                [0, 1, 0, 0, 19],
                [1, 0, 0, 0, 23],
                [0, 0, 0, 1, 25],
                [0, 0, 1, 0, 30],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ],
            [
                [1, 0, 0, 0, 35],
                [0, 0, 0, 1, 40],
                [0, 0, 1, 0, 12],
                [0, 1, 0, 0, 10],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0]
            ]
        ])
        np.testing.assert_array_equal(encoded_vectors, expected_vectors)

    def test_decode_char(self):
        encoder = OneHotMatrixEncoder(1)
        encoded_sequences = np.array([
            [
                [1, 0, 0, 0]
            ],
            [
                [0, 0, 0, 1]
            ]
        ])
        decoded_sequences = encoder.decode_sequences(encoded_sequences)
        expected_sequences = [
            ["A"],
            ["T"]
        ]
        np.testing.assert_array_equal(decoded_sequences, expected_sequences)

    def test_decode_sequences(self):
        encoder = OneHotMatrixEncoder(4)
        encoded_sequences = np.array([
            [
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0]
            ],
            [
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
                [0, 1, 0, 0]
            ]
        ])
        decoded_sequences = encoder.decode_sequences(encoded_sequences)
        expected_sequences = [
            ["C", "A", "T", "G"],
            ["A", "T", "G", "C"]
        ]
        np.testing.assert_array_equal(decoded_sequences, expected_sequences)

