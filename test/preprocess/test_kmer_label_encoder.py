from unittest import TestCase

import numpy as np

from preprocess.KmerLabelEncoder import KmerLabelEncoder


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

        input_matrix, output_matrix, shifted_output_matrix = \
            self.encoder.encode_kmers(input_kmers, output_kmers)

        self.assertEqual(input_matrix.shape, (2, 4))
        self.assertEqual(output_matrix.shape, (2, 2))
        self.assertEqual(shifted_output_matrix.shape, (2, 2))
        expected_input = np.array([
            [0, 0, 3, 3],
            [0, 3, 3, 2]
        ])
        expected_output = np.array([
            [3, 1],
            [1, 2]
        ])
        expected_shifted_output = np.array([
            [4, 3],
            [4, 1]
        ])
        np.testing.assert_array_equal(input_matrix, expected_input)
        np.testing.assert_array_equal(output_matrix, expected_output)
        np.testing.assert_array_equal(shifted_output_matrix, expected_shifted_output)

    def test_encode_kmers_no_shifted_output(self):
        input_kmers = [
            "AATT",
            "ATTG"
        ]
        output_kmers = [
            "TC",
            "CG"
        ]

        input_matrix, output_matrix, shifted_output_matrix = \
            self.encoder.encode_kmers(input_kmers, output_kmers, with_shifted_output=False)

        self.assertEqual(input_matrix.shape, (2, 4))
        self.assertEqual(output_matrix.shape, (2, 2))
        self.assertEqual(shifted_output_matrix.shape, (0,))
        expected_input = np.array([
            [0, 0, 3, 3],
            [0, 3, 3, 2]
        ])
        expected_output = np.array([
            [3, 1],
            [1, 2]
        ])
        expected_shifted_output = np.array([])
        np.testing.assert_array_equal(input_matrix, expected_input)
        np.testing.assert_array_equal(output_matrix, expected_output)
        np.testing.assert_array_equal(shifted_output_matrix, expected_shifted_output)

    def test_encode_kmers_no_input(self):
        input_kmers = []
        output_kmers = [
            "TC",
            "CG"
        ]

        input_matrix, output_matrix, shifted_output_matrix = \
            self.encoder.encode_kmers(input_kmers, output_kmers)

        self.assertEqual(input_matrix.shape, (0,))
        self.assertEqual(output_matrix.shape, (2, 2))
        self.assertEqual(shifted_output_matrix.shape, (2, 2))
        expected_input = np.array([])
        expected_output = np.array([
            [3, 1],
            [1, 2]
        ])
        expected_shifted_output = np.array([
            [4, 3],
            [4, 1]
        ])
        np.testing.assert_array_equal(input_matrix, expected_input)
        np.testing.assert_array_equal(output_matrix, expected_output)
        np.testing.assert_array_equal(shifted_output_matrix, expected_shifted_output)

    def test_encode_kmers_no_output(self):
        input_kmers = [
            "AATT",
            "ATTG"
        ]
        output_kmers = []

        input_matrix, output_matrix, shifted_output_matrix = \
            self.encoder.encode_kmers(input_kmers, output_kmers)

        self.assertEqual(input_matrix.shape, (2, 4))
        self.assertEqual(output_matrix.shape, (0,))
        self.assertEqual(shifted_output_matrix.shape, (0,))
        expected_input = np.array([
            [0, 0, 3, 3],
            [0, 3, 3, 2]
        ])
        expected_output = np.array([])
        expected_shifted_output = np.array([])
        np.testing.assert_array_equal(input_matrix, expected_input)
        np.testing.assert_array_equal(output_matrix, expected_output)
        np.testing.assert_array_equal(shifted_output_matrix, expected_shifted_output)