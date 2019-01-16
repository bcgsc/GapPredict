from unittest import TestCase

import numpy as np

from preprocess.VariableLengthKmerLabelEncoder import VariableLengthKmerLabelEncoder


class TestVariableLengthKmerLabelEncoder(TestCase):
    def setUp(self):
        self.encoder = VariableLengthKmerLabelEncoder()

    def test_encode_kmers(self):
        input_kmers = [
            "A",
            "G",
            "C",
            "T",
            "AA",
            "TG",
            "ACT",
            "TTG",
            "ATTG",
            "AATTG"
        ]
        output_kmers = [
            "TC",
            "TG",
            "GA",
            "AA",
            "GC",
            "GT",
            "CA",
            "TG",
            "GA",
            "CC"
        ]

        input_matrix, output_matrix, k_high = \
            self.encoder.encode_kmers(input_kmers, output_kmers)

        self.assertEqual(input_matrix.shape, (10, 5))
        self.assertEqual(output_matrix.shape, (10, 2))
        expected_input = np.array([
            [1, 0, 0, 0, 0],
            [3, 0, 0, 0, 0],
            [2, 0, 0, 0, 0],
            [4, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [4, 3, 0, 0, 0],
            [1, 2, 4, 0, 0],
            [4, 4, 3, 0, 0],
            [1, 4, 4, 3, 0],
            [1, 1, 4, 4, 3]
        ])
        expected_output = np.array([
            [4, 2],
            [4, 3],
            [3, 1],
            [1, 1],
            [3, 2],
            [3, 4],
            [2, 1],
            [4, 3],
            [3, 1],
            [2, 2]
        ])
        np.testing.assert_array_equal(input_matrix, expected_input)
        np.testing.assert_array_equal(output_matrix, expected_output)
        self.assertEqual(k_high, 5)

    def test_encode_kmers_no_shifted_output(self):
        input_kmers = [
            "A",
            "G",
            "C",
            "T",
            "AA",
            "TG",
            "ACT",
            "TTG",
            "ATTG",
            "AATTG"
        ]
        output_kmers = [
            "TC",
            "TG",
            "GA",
            "AA",
            "GC",
            "GT",
            "CA",
            "TG",
            "GA",
            "CC"
        ]

        input_matrix, output_matrix, k_high = \
            self.encoder.encode_kmers(input_kmers, output_kmers)

        self.assertEqual(input_matrix.shape, (10, 5))
        self.assertEqual(output_matrix.shape, (10, 2))
        expected_input = np.array([
            [1, 0, 0, 0, 0],
            [3, 0, 0, 0, 0],
            [2, 0, 0, 0, 0],
            [4, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [4, 3, 0, 0, 0],
            [1, 2, 4, 0, 0],
            [4, 4, 3, 0, 0],
            [1, 4, 4, 3, 0],
            [1, 1, 4, 4, 3]
        ])
        expected_output = np.array([
            [4, 2],
            [4, 3],
            [3, 1],
            [1, 1],
            [3, 2],
            [3, 4],
            [2, 1],
            [4, 3],
            [3, 1],
            [2, 2]
        ])
        np.testing.assert_array_equal(input_matrix, expected_input)
        np.testing.assert_array_equal(output_matrix, expected_output)
        self.assertEqual(k_high, 5)

    def test_encode_kmers_no_input(self):
        input_kmers = []
        output_kmers = [
            "TC",
            "TG",
            "GA",
            "AA",
            "GC",
            "GT",
            "CA",
            "TG",
            "GA",
            "CC"
        ]

        input_matrix, output_matrix, k_high = \
            self.encoder.encode_kmers(input_kmers, output_kmers)

        self.assertEqual(input_matrix.shape, (0,))
        self.assertEqual(output_matrix.shape, (10, 2))
        expected_input = np.array([])
        expected_output = np.array([
            [4, 2],
            [4, 3],
            [3, 1],
            [1, 1],
            [3, 2],
            [3, 4],
            [2, 1],
            [4, 3],
            [3, 1],
            [2, 2]
        ])
        np.testing.assert_array_equal(input_matrix, expected_input)
        np.testing.assert_array_equal(output_matrix, expected_output)
        self.assertEqual(k_high, 0)

    def test_encode_kmers_no_output(self):
        input_kmers = [
            "A",
            "G",
            "C",
            "T",
            "AA",
            "TG",
            "ACT",
            "TTG",
            "ATTG",
            "AATTG"
        ]
        output_kmers = []

        input_matrix, output_matrix, k_high = \
            self.encoder.encode_kmers(input_kmers, output_kmers)

        self.assertEqual(input_matrix.shape, (10, 5))
        self.assertEqual(output_matrix.shape, (0,))
        expected_input = np.array([
            [1, 0, 0, 0, 0],
            [3, 0, 0, 0, 0],
            [2, 0, 0, 0, 0],
            [4, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [4, 3, 0, 0, 0],
            [1, 2, 4, 0, 0],
            [4, 4, 3, 0, 0],
            [1, 4, 4, 3, 0],
            [1, 1, 4, 4, 3]
        ])
        expected_output = np.array([])
        np.testing.assert_array_equal(input_matrix, expected_input)
        np.testing.assert_array_equal(output_matrix, expected_output)
        self.assertEqual(k_high, 5)
