from unittest import TestCase

import numpy as np

from preprocess.SequenceComplementCalculator import SequenceComplementCalculator


class TestSequenceComplementCalculator(TestCase):
    def setUp(self):
        self.complement_calculator = SequenceComplementCalculator()

    def test_compare_no_sequences(self):
        np.testing.assert_array_equal(np.array([[]]), self.complement_calculator.compare_sequences(
            [],
            []
        ))

    def test_compare_non_complementary_sequences_strings(self):
        np.testing.assert_array_equal(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), self.complement_calculator.compare_sequences(
            ["ATGCATGCAT"],
            ["ATGCATGCAT"]
        ))

    def test_compare_with_empty_sequence_strings(self):
        np.testing.assert_array_equal(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), self.complement_calculator.compare_sequences(
            ["ATGCATGCAT"],
            [""]
        ))

    def test_compare_complementary_sequences_strings(self):
        np.testing.assert_array_equal(np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]), self.complement_calculator.compare_sequences(
            ["ATGCATGCAT"],
            ["TACGTACGTA"]
        ))

    def test_compare_partly_complementary_sequences_strings(self):
        np.testing.assert_array_equal(np.array([[1, 0, 1, 0, 1, 1, 0, 1, 0, 0]]), self.complement_calculator.compare_sequences(
            ["ATGCATGCAT"],
            ["TTCCTAGGAT"]
        ))

    def test_compare_non_complementary_sequences_from_offset_strings(self):
        np.testing.assert_array_equal(np.array([[0, 0, 0, 0, 0, 0, 0]]), self.complement_calculator.compare_sequences(
            ["ATGCATGCAT"],
            ["ATGCATGCAT"],
            3
        ))

    def test_compare_non_complementary_sequences_from_offset_longer_than_string_strings(self):
        np.testing.assert_array_equal(np.array([[]]), self.complement_calculator.compare_sequences(
            ["ATGCATGCAT"],
            ["ATGCATGCAT"],
            11
        ))

    def test_compare_non_complementary_sequences_from_negative_offset_strings(self):
        np.testing.assert_array_equal(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), self.complement_calculator.compare_sequences(
            ["ATGCATGCAT"],
            ["ATGCATGCAT"],
            -2
        ))

    def test_compare_complementary_sequences_from_offset_strings(self):
        np.testing.assert_array_equal(np.array([[1, 1, 1, 1, 1, 1, 1]]), self.complement_calculator.compare_sequences(
            ["ATGCATGCAT"],
            ["TACGTACGTA"],
            3
        ))

    def test_compare_partly_complementary_sequences_from_offset_strings(self):
        np.testing.assert_array_equal(np.array([[0, 1, 1, 0, 1, 0, 0]]), self.complement_calculator.compare_sequences(
            ["ATGCATGCAT"],
            ["TTCCTAGGAT"],
            3
        ))

    def test_compare_partly_complementary_sequences_from_offset_check_4_bases_strings(self):
        np.testing.assert_array_equal(np.array([[0, 1, 1, 0]]), self.complement_calculator.compare_sequences(
            ["ATGCATGCAT"],
            ["TTCCTAGGAT"],
            3,
            4
        ))

    def test_compare_partly_complementary_sequences_from_offset_check_20_bases_doesnt_exceed_strings(self):
        np.testing.assert_array_equal(np.array([[0, 1, 1, 0, 1, 0, 0]]), self.complement_calculator.compare_sequences(
            ["ATGCATGCAT"],
            ["TTCCTAGGAT"],
            3,
            20
        ))

    def test_compare_multiple_partly_complementary_sequences_from_offset_check_20_bases_doesnt_exceed_strings(self):
        output = self.complement_calculator.compare_sequences(
            ["ATGCATGCAT",
             "TTCCTAGGAT"],
            ["TTCCTAGGAT",
             "AAGGATCCAA"],
            3,
            20
        )
        np.testing.assert_array_equal(np.array([[0, 1, 1, 0, 1, 0, 0],
                                                [1, 1, 1, 1, 1, 0, 1]]), output)

    def test_compare_partly_complementary_sequences_from_offset_check_negative_bases_treated_as_0_strings(self):
        np.testing.assert_array_equal(np.array([[]]), self.complement_calculator.compare_sequences(
            ["ATGCATGCAT"],
            ["TTCCTAGGAT"],
            3,
            -5
        ))

    def test_compare_non_complementary_sequences(self):
        np.testing.assert_array_equal(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), self.complement_calculator.compare_sequences(
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"]],
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"]]
        ))

    def test_compare_with_empty_sequence(self):
        np.testing.assert_array_equal(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), self.complement_calculator.compare_sequences(
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"]],
            [[]]
        ))

    def test_compare_with_wild_card_sequence(self):
        np.testing.assert_array_equal(np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]), self.complement_calculator.compare_sequences(
            [["N", "N", "N", "N", "N", "N", "N", "N", "N", "N"]],
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"]]
        ))

    def test_compare_with_partial_wild_card_sequence(self):
        np.testing.assert_array_equal(np.array([[1, 0, 1, 0, 1, 1, 0, 1, 1, 1]]), self.complement_calculator.compare_sequences(
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"]],
            [["T", "G", "C", "A", "T", "A", "G", "G", "N", "N"]]
        ))

    def test_compare_complementary_sequences(self):
        np.testing.assert_array_equal(np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]), self.complement_calculator.compare_sequences(
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"]],
            [["T", "A", "C", "G", "T", "A", "C", "G", "T", "A"]]
        ))

    def test_compare_partly_complementary_sequences(self):
        np.testing.assert_array_equal(np.array([[1, 0, 1, 0, 1, 1, 0, 1, 0, 0]]), self.complement_calculator.compare_sequences(
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"]],
            [["T", "G", "C", "A", "T", "A", "G", "G", "A", "T"]]
        ))

    def test_compare_non_complementary_sequences_from_offset(self):
        np.testing.assert_array_equal(np.array([[0, 0, 0, 0, 0, 0, 0]]), self.complement_calculator.compare_sequences(
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"]],
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"]],
            3
        ))

    def test_compare_non_complementary_sequences_from_offset_longer_than_string(self):
        np.testing.assert_array_equal(np.array([[]]), self.complement_calculator.compare_sequences(
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"]],
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"]],
            11
        ))

    def test_compare_non_complementary_sequences_from_negative_offset(self):
        np.testing.assert_array_equal(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), self.complement_calculator.compare_sequences(
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"]],
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"]],
            -2
        ))

    def test_compare_complementary_sequences_from_offset(self):
        np.testing.assert_array_equal(np.array([[1, 1, 1, 1, 1, 1, 1]]), self.complement_calculator.compare_sequences(
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"]],
            [["T", "A", "C", "G", "T", "A", "C", "G", "T", "A"]],
            3
        ))

    def test_compare_partly_complementary_sequences_from_offset(self):
        np.testing.assert_array_equal(np.array([[0, 1, 1, 0, 1, 0, 0]]), self.complement_calculator.compare_sequences(
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"]],
            [["T", "G", "C", "A", "T", "A", "G", "G", "A", "T"]],
            3
        ))

    def test_compare_partly_complementary_sequences_from_offset_check_4_bases(self):
        np.testing.assert_array_equal(np.array([[0, 1, 1, 0]]), self.complement_calculator.compare_sequences(
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"]],
            [["T", "G", "C", "A", "T", "A", "G", "G", "A", "T"]],
            3,
            4
        ))

    def test_compare_partly_complementary_sequences_from_offset_check_20_bases_doesnt_exceed(self):
        np.testing.assert_array_equal(np.array([[0, 1, 1, 0, 1, 0, 0]]), self.complement_calculator.compare_sequences(
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"]],
            [["T", "G", "C", "A", "T", "A", "G", "G", "A", "T"]],
            3,
            20
        ))

    def test_compare_multiple_partly_complementary_sequences_from_offset_check_20_bases_doesnt_exceed(self):
        output = self.complement_calculator.compare_sequences(
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"],
             ["T", "T", "C", "C", "T", "A", "G", "G", "A", "T"]],
            [["T", "T", "C", "C", "T", "A", "G", "G", "A", "T"],
             ["A", "A", "G", "G", "A", "T", "C", "C", "A", "A"]],
            3,
            20 
        )
        np.testing.assert_array_equal(np.array([[0, 1, 1, 0, 1, 0, 0],
                                                [1, 1, 1, 1, 1, 0, 1]]), output)

    def test_compare_partly_complementary_sequences_from_offset_check_negative_bases_treated_as_0(self):
        np.testing.assert_array_equal(np.array([[]]), self.complement_calculator.compare_sequences(
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"]],
            [["A", "G", "G", "A", "A", "T", "C", "C", "G", "A"]],
            3,
            -5
        ))
