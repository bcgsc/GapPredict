from unittest import TestCase

import numpy as np

from preprocess.SequenceMatchCalculator import SequenceMatchCalculator


class TestSequenceMatchCalculator(TestCase):
    def setUp(self):
        self.matcher = SequenceMatchCalculator()

    def test_compare_no_sequences(self):
        np.testing.assert_array_equal(np.array([[]]), self.matcher.compare_sequences(
            [],
            []
        ))

    def test_compare_different_sequences_strings(self):
        np.testing.assert_array_equal(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), self.matcher.compare_sequences(
            ["ATGCATGCAT"],
            ["TGCATGCATG"]
        ))

    def test_compare_with_empty_sequence_strings(self):
        np.testing.assert_array_equal(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), self.matcher.compare_sequences(
            ["ATGCATGCAT"],
            [""]
        ))

    def test_compare_same_sequences_strings(self):
        np.testing.assert_array_equal(np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]), self.matcher.compare_sequences(
            ["ATGCATGCAT"],
            ["ATGCATGCAT"]
        ))

    def test_compare_similar_sequences_strings(self):
        np.testing.assert_array_equal(np.array([[1, 0, 1, 0, 1, 1, 0, 1, 0, 0]]), self.matcher.compare_sequences(
            ["ATGCATGCAT"],
            ["AGGAATCCGA"]
        ))

    def test_compare_different_sequences_from_offset_strings(self):
        np.testing.assert_array_equal(np.array([[0, 0, 0, 0, 0, 0, 0]]), self.matcher.compare_sequences(
            ["ATGCATGCAT"],
            ["TGCATGCATG"],
            3
        ))

    def test_compare_different_sequences_from_offset_longer_than_string_strings(self):
        np.testing.assert_array_equal(np.array([[]]), self.matcher.compare_sequences(
            ["ATGCATGCAT"],
            ["TGCATGCATG"],
            11
        ))

    def test_compare_different_sequences_from_negative_offset_strings(self):
        np.testing.assert_array_equal(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), self.matcher.compare_sequences(
            ["ATGCATGCAT"],
            ["TGCATGCATG"],
            -2
        ))

    def test_compare_same_sequences_from_offset_strings(self):
        np.testing.assert_array_equal(np.array([[1, 1, 1, 1, 1, 1, 1]]), self.matcher.compare_sequences(
            ["ATGCATGCAT"],
            ["ATGCATGCAT"],
            3
        ))

    def test_compare_similar_sequences_from_offset_strings(self):
        np.testing.assert_array_equal(np.array([[0, 1, 1, 0, 1, 0, 0]]), self.matcher.compare_sequences(
            ["ATGCATGCAT"],
            ["AGGAATCCGA"],
            3
        ))

    def test_compare_similar_sequences_from_offset_check_4_bases_strings(self):
        np.testing.assert_array_equal(np.array([[0, 1, 1, 0]]), self.matcher.compare_sequences(
            ["ATGCATGCAT"],
            ["AGGAATCCGA"],
            3,
            4
        ))

    def test_compare_similar_sequences_from_offset_check_20_bases_doesnt_exceed_strings(self):
        np.testing.assert_array_equal(np.array([[0, 1, 1, 0, 1, 0, 0]]), self.matcher.compare_sequences(
            ["ATGCATGCAT"],
            ["AGGAATCCGA"],
            3,
            20
        ))

    def test_compare_multiple_similar_sequences_from_offset_check_20_bases_doesnt_exceed_strings(self):
        output = self.matcher.compare_sequences(
            ["ATGCATGCAT",
             "AAAGCTGAAT"],
            ["AGGAATCCGA",
             "AGGGCTGAGT"],
            3,
            20
        )
        np.testing.assert_array_equal(np.array([[0, 1, 1, 0, 1, 0, 0],
                                                [1, 1, 1, 1, 1, 0, 1]]), output)

    def test_compare_similar_sequences_from_offset_check_negative_bases_treated_as_0_strings(self):
        np.testing.assert_array_equal(np.array([[]]), self.matcher.compare_sequences(
            ["ATGCATGCAT"],
            ["AGGAATCCGA"],
            3,
            -5
        ))

    def test_compare_different_sequences(self):
        np.testing.assert_array_equal(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), self.matcher.compare_sequences(
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"]],
            [["T", "G", "C", "A", "T", "G", "C", "A", "T", "G"]]
        ))

    def test_compare_with_empty_sequence(self):
        np.testing.assert_array_equal(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), self.matcher.compare_sequences(
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"]],
            [[]]
        ))

    def test_compare_with_wild_card_sequence(self):
        np.testing.assert_array_equal(np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]), self.matcher.compare_sequences(
            [["N", "N", "N", "N", "N", "N", "N", "N", "N", "N"]],
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"]]
        ))

    def test_compare_with_partial_wild_card_sequence(self):
        np.testing.assert_array_equal(np.array([[1, 0, 1, 0, 1, 1, 0, 1, 1, 1]]), self.matcher.compare_sequences(
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"]],
            [["A", "G", "G", "A", "A", "T", "C", "C", "N", "N"]]
        ))

    def test_compare_same_sequences(self):
        np.testing.assert_array_equal(np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]), self.matcher.compare_sequences(
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"]],
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"]]
        ))

    def test_compare_similar_sequences(self):
        np.testing.assert_array_equal(np.array([[1, 0, 1, 0, 1, 1, 0, 1, 0, 0]]), self.matcher.compare_sequences(
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"]],
            [["A", "G", "G", "A", "A", "T", "C", "C", "G", "A"]]
        ))

    def test_compare_different_sequences_from_offset(self):
        np.testing.assert_array_equal(np.array([[0, 0, 0, 0, 0, 0, 0]]), self.matcher.compare_sequences(
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"]],
            [["T", "G", "C", "A", "T", "G", "C", "A", "T", "G"]],
            3
        ))

    def test_compare_different_sequences_from_offset_longer_than_string(self):
        np.testing.assert_array_equal(np.array([[]]), self.matcher.compare_sequences(
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"]],
            [["T", "G", "C", "A", "T", "G", "C", "A", "T", "G"]],
            11
        ))

    def test_compare_different_sequences_from_negative_offset(self):
        np.testing.assert_array_equal(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), self.matcher.compare_sequences(
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"]],
            [["T", "G", "C", "A", "T", "G", "C", "A", "T", "G"]],
            -2
        ))

    def test_compare_same_sequences_from_offset(self):
        np.testing.assert_array_equal(np.array([[1, 1, 1, 1, 1, 1, 1]]), self.matcher.compare_sequences(
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"]],
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"]],
            3
        ))

    def test_compare_similar_sequences_from_offset(self):
        np.testing.assert_array_equal(np.array([[0, 1, 1, 0, 1, 0, 0]]), self.matcher.compare_sequences(
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"]],
            [["A", "G", "G", "A", "A", "T", "C", "C", "G", "A"]],
            3
        ))

    def test_compare_similar_sequences_from_offset_check_4_bases(self):
        np.testing.assert_array_equal(np.array([[0, 1, 1, 0]]), self.matcher.compare_sequences(
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"]],
            [["A", "G", "G", "A", "A", "T", "C", "C", "G", "A"]],
            3,
            4
        ))

    def test_compare_similar_sequences_from_offset_check_20_bases_doesnt_exceed(self):
        np.testing.assert_array_equal(np.array([[0, 1, 1, 0, 1, 0, 0]]), self.matcher.compare_sequences(
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"]],
            [["A", "G", "G", "A", "A", "T", "C", "C", "G", "A"]],
            3,
            20
        ))

    def test_compare_multiple_similar_sequences_from_offset_check_20_bases_doesnt_exceed(self):
        output = self.matcher.compare_sequences(
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"],
             ["A", "A", "A", "G", "C", "T", "G", "A", "A", "T"]],
            [["A", "G", "G", "A", "A", "T", "C", "C", "G", "A"],
             ["A", "G", "G", "G", "C", "T", "G", "A", "G", "T"]],
            3,
            20
        )
        np.testing.assert_array_equal(np.array([[0, 1, 1, 0, 1, 0, 0],
                                                [1, 1, 1, 1, 1, 0, 1]]), output)

    def test_compare_similar_sequences_from_offset_check_negative_bases_treated_as_0(self):
        np.testing.assert_array_equal(np.array([[]]), self.matcher.compare_sequences(
            [["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"]],
            [["A", "G", "G", "A", "A", "T", "C", "C", "G", "A"]],
            3,
            -5
        ))
