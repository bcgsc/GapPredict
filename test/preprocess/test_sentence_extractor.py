from unittest import TestCase
from preprocess.SentenceExtractor import SentenceExtractor
from exceptions.NonpositiveLengthException import NonpositiveLengthException
from models.ParsedFastqRecord import ParsedFastqRecord

import numpy as np

class TestSentenceExtractor(TestCase):
    def setUp(self):
        self.extractor = SentenceExtractor()
        self.sequences = [
            ParsedFastqRecord("AATTGAGTCG", np.array([28, 28, 28, 10, 28, 32, 32, 28, 32, 35])),
            ParsedFastqRecord("AGTCGAATTG", np.array([28, 28, 28, 10, 28, 32, 32, 28, 32, 35])),
            ParsedFastqRecord("TCGAATTGAG", np.array([28, 28, 28, 10, 28, 32, 32, 28, 32, 35]))
        ]
        self.sequences_with_errors = [
            ParsedFastqRecord("AATTGAGTCG", np.array([28, 28, 28, 10, 28, 32, 32, 28, 32, 35])),
            ParsedFastqRecord("AGNCGAATTG", np.array([28, 28, 28, 10, 28, 32, 32, 28, 32, 35])),
            ParsedFastqRecord("TCGAATTGNG", np.array([28, 28, 28, 10, 28, 32, 32, 28, 32, 35]))
        ]

    def test_split_sequences_into_kmers_k_zero(self):
        try:
            sentences = self.extractor.split_sequences_into_kmers(self.sequences, 0)
        except NonpositiveLengthException as e:
            pass
        except Exception as e:
            self.fail()

    def test_split_sequences_into_kmers_k_negative(self):
        try:
            sentences = self.extractor.split_sequences_into_kmers(self.sequences, -1)
        except NonpositiveLengthException as e:
            pass
        except Exception as e:
            self.fail()

    def test_split_sequences_into_kmers(self):
        sentences = self.extractor.split_sequences_into_kmers(self.sequences, 5)
        expected_sentences = [
            ["AATTG", "ATTGA", "TTGAG", "TGAGT", "GAGTC", "AGTCG"],
            ["AGTCG", "GTCGA", "TCGAA", "CGAAT", "GAATT", "AATTG"],
            ["TCGAA", "CGAAT", "GAATT", "AATTG", "ATTGA", "TTGAG"]
        ]
        np.testing.assert_array_equal(sentences, expected_sentences)

    def test_split_sequences_into_bases(self):
        sentences = self.extractor.split_sequences_into_kmers(self.sequences, 1)
        expected_sentences = [
            ["A", "A", "T", "T", "G", "A", "G", "T", "C", "G"],
            ["A", "G", "T", "C", "G", "A", "A", "T", "T", "G"],
            ["T", "C", "G", "A", "A", "T", "T", "G", "A", "G"]
        ]
        np.testing.assert_array_equal(sentences, expected_sentences)

    def test_split_sequences_into_kmers_k_too_long(self):
        sentences = self.extractor.split_sequences_into_kmers(self.sequences, 11)
        expected_sentences = [
            ["AATTGAGTCG"],
            ["AGTCGAATTG"],
            ["TCGAATTGAG"]
        ]
        np.testing.assert_array_equal(sentences, expected_sentences)

    def test_split_sequences_into_kmers_with_errors(self):
        sentences = self.extractor.split_sequences_into_kmers(self.sequences_with_errors, 5)
        expected_sentences = [
            ["AATTG", "ATTGA", "TTGAG", "TGAGT", "GAGTC", "AGTCG"]
        ]
        np.testing.assert_array_equal(sentences, expected_sentences)
