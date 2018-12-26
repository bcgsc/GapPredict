from unittest import TestCase

from preprocess.SequenceReverser import SequenceReverser
from models.ParsedFastqRecord import ParsedFastqRecord


class TestSequenceReverser(TestCase):
    def setUp(self):
        self.reverser = SequenceReverser()

    def test_reverse_complement_with_n(self):
        reverse_complement = self.reverser.reverse_complement("ATCGNT")
        self.assertEqual(reverse_complement, "ANCGAT")

    def test_reverse_complement(self):
        reverse_complement = self.reverser.reverse_complement("ATGCCGT")
        self.assertEqual(reverse_complement, "ACGGCAT")

    def test_reverse_complement_empty(self):
        reverse_complement = self.reverser.reverse_complement("")
        self.assertEqual(reverse_complement, "")

    def test_reverse_sequence(self):
        sequence = ParsedFastqRecord("ATGCCGT", [10, 11, 12, 13, 14, 15, 16])
        reverse_sequence = self.reverser.reverse_sequence(sequence)
        expected_reverse_sequence = ParsedFastqRecord("ACGGCAT", [16, 15, 14, 13, 12, 11, 10])
        self.assertEqual(reverse_sequence, expected_reverse_sequence)

    def test_reverse_sequence_empty(self):
        sequence = ParsedFastqRecord("", [])
        reverse_sequence = self.reverser.reverse_sequence(sequence)
        expected_reverse_sequence = ParsedFastqRecord("", [])
        self.assertEqual(reverse_sequence, expected_reverse_sequence)

    def test_reverse_sequence_palindrome(self):
        sequence = ParsedFastqRecord("ATGCAT", [10, 11, 12, 13, 14, 15])
        reverse_sequence = self.reverser.reverse_sequence(sequence)
        expected_reverse_sequence = ParsedFastqRecord("ATGCAT", [15, 14, 13, 12, 11, 10])
        self.assertEqual(reverse_sequence, expected_reverse_sequence)
