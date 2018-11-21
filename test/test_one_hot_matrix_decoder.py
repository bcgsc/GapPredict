from unittest import TestCase

import numpy as np

from exceptions.NonpositiveLengthException import NonpositiveLengthException
from onehot.OneHotMatrix import OneHotMatrixDecoder


class TestOneHotMatrixDecoder(TestCase):
    def test_encode_bad_length(self):
        try:
            decoder = OneHotMatrixDecoder(0)
        except NonpositiveLengthException as e:
            pass
        except Exception as e:
            self.fail()

    def test_decode_sequences(self):
        self.fail()

    def test_decode_char(self):
        decoder = OneHotMatrixDecoder(1)
        encoded_sequences = np.array([
            [
                [1, 0, 0, 0, 0]
            ],
            [
                [0, 0, 0, 1, 0]
            ],
            [
                [0, 0, 0, 0, 1]
            ]
        ])
        decoded_sequences = decoder.decode_sequences(encoded_sequences)
        expected_sequences = [
            ["A"],
            ["T"],
            ["!"]
        ]
        np.testing.assert_array_equal(decoded_sequences, expected_sequences)

    def test_decode_sequences(self):
        decoder = OneHotMatrixDecoder(4)
        encoded_sequences = np.array([
            [
                [0, 1, 0, 0, 0],
                [1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0]
            ],
            [
                [1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0]
            ],
            [
                [1, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0]
            ]
        ])
        decoded_sequences = decoder.decode_sequences(encoded_sequences)
        expected_sequences = [
            ["C", "A", "T", "G"],
            ["A", "T", "G", "C"],
            ["A", "!", "G", "C"]
        ]
        np.testing.assert_array_equal(decoded_sequences, expected_sequences)