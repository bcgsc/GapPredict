from unittest import TestCase

from SequenceMatchCalculator import SequenceMatchCalculator


class TestSequenceMatchCalculator(TestCase):
    def setUp(self):
        self.matcher = SequenceMatchCalculator()

    def test_compare_different_sequences(self):
        self.assertEqual(10, self.matcher.compare_sequences("ATGCATGCAT", "TGCATGCATG"))

    def test_compare_same_sequences(self):
        self.assertEqual(0, self.matcher.compare_sequences("ATGCATGCAT", "ATGCATGCAT"))

    def test_compare_similar_sequences(self):
        self.assertEqual(5, self.matcher.compare_sequences("ATGCATGCAT", "AGGAATCCGA"))

