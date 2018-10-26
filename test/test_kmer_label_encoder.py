from unittest import TestCase

import numpy as np

from KmerLabelEncoder import KmerLabelEncoder


class TestKmerLabelEncoder(TestCase):
    def setUp(self):
        self.encoder = KmerLabelEncoder()

    def test_encode_kmers(self):
        input_kmers = [
            "AATT",
            "ATTG"
        ]
        output_kmers = [
            "TC",
            "CG"
        ]
        input_quality = [
            [28, 28, 28, 10],
            [28, 28, 10, 28]
        ]

        input_matrix, input_quality_matrix, output_matrix, shifted_output_matrix = \
            self.encoder.encode_kmers(input_kmers, output_kmers, input_quality)

        self.assertEqual(input_matrix.shape, (2, 4))
        self.assertEqual(input_quality_matrix.shape, (2, 4))
        self.assertEqual(output_matrix.shape, (2, 2))
        self.assertEqual(shifted_output_matrix.shape, (2, 2))
        expected_input = np.array([
            [1, 1, 4, 4],
            [1, 4, 4, 3]
        ])
        expected_input_quality = np.array([
            [28, 28, 28, 10],
            [28, 28, 10, 28]
        ])
        expected_output = np.array([
            [4, 2],
            [2, 3]
        ])
        expected_shifted_output = np.array([
            [0, 4],
            [0, 2]
        ])
        np.testing.assert_array_equal(input_matrix, expected_input)
        np.testing.assert_array_equal(input_quality_matrix, expected_input_quality)
        np.testing.assert_array_equal(output_matrix, expected_output)
        np.testing.assert_array_equal(shifted_output_matrix, expected_shifted_output)

    def test_encode_kmers_fill_in_the_blanks(self):
        input_kmers = [
            "AATT",
            "ATTG"
        ]
        output_kmers = [
            "TC",
            "CG"
        ]
        input_quality = [
            [28, 28, 28, 10],
            [28, 28, 10, 28]
        ]

        input_matrix, input_quality_matrix, output_matrix, shifted_output_matrix = \
            self.encoder.encode_kmers(input_kmers, output_kmers, input_quality, fill_in_the_blanks=True)

        self.assertEqual(input_matrix.shape, (2, 4))
        self.assertEqual(input_quality_matrix.shape, (2, 4))
        self.assertEqual(output_matrix.shape, (2, 6))
        self.assertEqual(shifted_output_matrix.shape, (2, 6))
        expected_input = np.array([
            [1, 1, 4, 4],
            [1, 4, 4, 3]
        ])
        expected_input_quality = np.array([
            [28, 28, 28, 10],
            [28, 28, 10, 28]
        ])
        expected_output = np.array([
            [1, 1, 4, 4, 4, 2],
            [1, 4, 4, 3, 2, 3]
        ])
        expected_shifted_output = np.array([
            [0, 1, 1, 4, 4, 4],
            [0, 1, 4, 4, 3, 2]
        ])
        np.testing.assert_array_equal(input_matrix, expected_input)
        np.testing.assert_array_equal(input_quality_matrix, expected_input_quality)
        np.testing.assert_array_equal(output_matrix, expected_output)
        np.testing.assert_array_equal(shifted_output_matrix, expected_shifted_output)