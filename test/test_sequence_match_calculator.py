from unittest import TestCase

from SequenceMatchCalculator import SequenceMatchCalculator


class TestSequenceMatchCalculator(TestCase):
    def setUp(self):
        self.matcher = SequenceMatchCalculator()

    def test_compare_different_sequences(self):
        self.assertEqual(10, self.matcher.compare_sequences(
            ["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"],
            ["T", "G", "C", "A", "T", "G", "C", "A", "T", "G"]
        ))

    def test_compare_same_sequences(self):
        self.assertEqual(0, self.matcher.compare_sequences(
            ["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"],
            ["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"]
        ))

    def test_compare_similar_sequences(self):
        self.assertEqual(5, self.matcher.compare_sequences(
            ["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"],
            ["A", "G", "G", "A", "A", "T", "C", "C", "G", "A"]
        ))

    def test_compare_different_sequences_from_offset(self):
        self.assertEqual(7, self.matcher.compare_sequences(
            ["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"],
            ["T", "G", "C", "A", "T", "G", "C", "A", "T", "G"],
            3
        ))

    def test_compare_different_sequences_from_offset_longer_than_string(self):
        self.assertEqual(0, self.matcher.compare_sequences(
            ["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"],
            ["T", "G", "C", "A", "T", "G", "C", "A", "T", "G"],
            11
        ))

    def test_compare_different_sequences_from_negative_offset(self):
        self.assertEqual(10, self.matcher.compare_sequences(
            ["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"],
            ["T", "G", "C", "A", "T", "G", "C", "A", "T", "G"],
            -2
        ))

    def test_compare_same_sequences_from_offset(self):
        self.assertEqual(0, self.matcher.compare_sequences(
            ["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"],
            ["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"],
            3
        ))

    def test_compare_similar_sequences_from_offset(self):
        self.assertEqual(4, self.matcher.compare_sequences(
            ["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"],
            ["A", "G", "G", "A", "A", "T", "C", "C", "G", "A"],
            3
        ))

    def test_compare_similar_sequences_from_offset_check_4_bases(self):
        self.assertEqual(2, self.matcher.compare_sequences(
            ["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"],
            ["A", "G", "G", "A", "A", "T", "C", "C", "G", "A"],
            3,
            4
        ))

    def test_compare_similar_sequences_from_offset_check_20_bases_doesnt_exceed(self):
        self.assertEqual(4, self.matcher.compare_sequences(
            ["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"],
            ["A", "G", "G", "A", "A", "T", "C", "C", "G", "A"],
            3,
            20
        ))

    def test_compare_similar_sequences_from_offset_check_negative_bases_treated_as_0(self):
        self.assertEqual(0, self.matcher.compare_sequences(
            ["A", "T", "G", "C", "A", "T", "G", "C", "A", "T"],
            ["A", "G", "G", "A", "A", "T", "C", "C", "G", "A"],
            3,
            -5
        ))
