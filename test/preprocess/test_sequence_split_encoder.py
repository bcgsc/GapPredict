from unittest import TestCase

import numpy as np

from preprocess.SequenceSplitEncoder import SequenceSplitEncoder


class TestSequenceSplitEncoder(TestCase):
    def setUp(self):
        self.uniform_sequences = np.array([
            "CTATTTGCTG",
            "GCTAATGCAG",
            "TGGGAAGTCT"
        ])
        self.nonuniform_sequences = np.array([
            "ATTT",
            "TGAGTG",
            "CGTATATT"
        ])

    def test_negative_split_idx(self):
        splitter = SequenceSplitEncoder(-1)
        input, output, shifted_output = splitter.split_sequences(self.nonuniform_sequences)
        expected_input = np.array([
            [0],
            [3],
            [1]
        ])
        expected_output = np.array([
            [3, 3, 3],
            [2, 0, 2],
            [2, 3, 0]
        ])
        expected_shifted_output = np.array([
            [4, 3, 3],
            [4, 2, 0],
            [4, 2, 3]
        ])
        np.testing.assert_array_equal(input, expected_input)
        np.testing.assert_array_equal(output, expected_output)
        np.testing.assert_array_equal(shifted_output, expected_shifted_output)


    def test_split_sequences_split_idx_at_zero(self):
        splitter = SequenceSplitEncoder(0)
        input, output, shifted_output = splitter.split_sequences(self.nonuniform_sequences)
        expected_input = np.array([
            [0],
            [3],
            [1]
        ])
        expected_output = np.array([
            [3, 3, 3],
            [2, 0, 2],
            [2, 3, 0]
        ])
        expected_shifted_output = np.array([
            [4, 3, 3],
            [4, 2, 0],
            [4, 2, 3]
        ])
        np.testing.assert_array_equal(input, expected_input)
        np.testing.assert_array_equal(output, expected_output)
        np.testing.assert_array_equal(shifted_output, expected_shifted_output)

    def test_split_sequences_split_idx_at_max_length(self):
        splitter = SequenceSplitEncoder(8)
        input, output, shifted_output = splitter.split_sequences(self.nonuniform_sequences)
        expected_input = np.array([
            [0, 3, 3],
            [3, 2, 0],
            [1, 2, 3]
        ])
        expected_output = np.array([
            [3],
            [2],
            [0]
        ])
        expected_shifted_output = np.array([
            [4],
            [4],
            [4]
        ])
        np.testing.assert_array_equal(input, expected_input)
        np.testing.assert_array_equal(output, expected_output)
        np.testing.assert_array_equal(shifted_output, expected_shifted_output)

    def test_split_sequences_split_idx_at_min_length(self):
        splitter = SequenceSplitEncoder(4)
        input, output, shifted_output = splitter.split_sequences(self.nonuniform_sequences)
        expected_input = np.array([
            [0, 3, 3],
            [3, 2, 0],
            [1, 2, 3]
        ])
        expected_output = np.array([
            [3],
            [2],
            [0]
        ])
        expected_shifted_output = np.array([
            [4],
            [4],
            [4]
        ])
        np.testing.assert_array_equal(input, expected_input)
        np.testing.assert_array_equal(output, expected_output)
        np.testing.assert_array_equal(shifted_output, expected_shifted_output)

    def test_split_sequences_split_idx_right_before_min_length(self):
        splitter = SequenceSplitEncoder(3)
        input, output, shifted_output = splitter.split_sequences(self.nonuniform_sequences)
        expected_input = np.array([
            [0, 3, 3],
            [3, 2, 0],
            [1, 2, 3]
        ])
        expected_output = np.array([
            [3],
            [2],
            [0]
        ])
        expected_shifted_output = np.array([
            [4],
            [4],
            [4]
        ])
        np.testing.assert_array_equal(input, expected_input)
        np.testing.assert_array_equal(output, expected_output)
        np.testing.assert_array_equal(shifted_output, expected_shifted_output)

    def test_split_sequences_split_idx_between_min_max_length(self):
        splitter = SequenceSplitEncoder(5)
        input, output, shifted_output = splitter.split_sequences(self.nonuniform_sequences)
        expected_input = np.array([
            [0, 3, 3],
            [3, 2, 0],
            [1, 2, 3]
        ])
        expected_output = np.array([
            [3],
            [2],
            [0]
        ])
        expected_shifted_output = np.array([
            [4],
            [4],
            [4]
        ])
        np.testing.assert_array_equal(input, expected_input)
        np.testing.assert_array_equal(output, expected_output)
        np.testing.assert_array_equal(shifted_output, expected_shifted_output)

    def test_split_sequences_split_idx_past_max_length(self):
        splitter = SequenceSplitEncoder(9)
        input, output, shifted_output = splitter.split_sequences(self.nonuniform_sequences)
        expected_input = np.array([
            [0, 3, 3],
            [3, 2, 0],
            [1, 2, 3]
        ])
        expected_output = np.array([
            [3],
            [2],
            [0]
        ])
        expected_shifted_output = np.array([
            [4],
            [4],
            [4]
        ])
        np.testing.assert_array_equal(input, expected_input)
        np.testing.assert_array_equal(output, expected_output)
        np.testing.assert_array_equal(shifted_output, expected_shifted_output)

    def test_split_sequences_uniform_lengths(self):
        splitter = SequenceSplitEncoder(7)
        input, output, shifted_output = splitter.split_sequences(self.uniform_sequences)
        expected_input = np.array([
            [1, 3, 0, 3, 3, 3, 2],
            [2, 1, 3, 0, 0, 3, 2],
            [3, 2, 2, 2, 0, 0, 2]
        ])
        expected_output = np.array([
            [1, 3, 2],
            [1, 0, 2],
            [3, 1, 3]
        ])
        expected_shifted_output = np.array([
            [4, 1, 3],
            [4, 1, 0],
            [4, 3, 1]
        ])
        np.testing.assert_array_equal(input, expected_input)
        np.testing.assert_array_equal(output, expected_output)
        np.testing.assert_array_equal(shifted_output, expected_shifted_output)

    def test_split_sequences_nonuniform_lengths(self):
        splitter = SequenceSplitEncoder(2)
        input, output, shifted_output = splitter.split_sequences(self.nonuniform_sequences)
        expected_input = np.array([
            [0, 3],
            [3, 2],
            [1, 2]
        ])
        expected_output = np.array([
            [3, 3],
            [0, 2],
            [3, 0]
        ])
        expected_shifted_output = np.array([
            [4, 3],
            [4, 0],
            [4, 3]
        ])
        np.testing.assert_array_equal(input, expected_input)
        np.testing.assert_array_equal(output, expected_output)
        np.testing.assert_array_equal(shifted_output, expected_shifted_output)