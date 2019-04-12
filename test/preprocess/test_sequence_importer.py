from unittest import TestCase

import numpy as np

from preprocess.BaseQualityConverter import BaseQualityConverter
from preprocess.SequenceImporter import SequenceImporter
from preprocess.SequenceReverser import SequenceReverser


class TestSequenceImporter(TestCase):
    def setUp(self):
        self.importer = SequenceImporter()
        self.converter = BaseQualityConverter()
        self.reverser = SequenceReverser()

    def test_import_fastq(self):
        sequences = self.importer.import_fastq(['../data/mini_read_for_test.fastq', '../data/mini_read_for_test_2.fastq.gz'])
        expected_sequences = np.array([
            "AATTGAGTCGTAGTATCCACACCAAGCCGGCGTTATCCGGTGAGGCGCAATGTTGCGGGGGCTTTATCCCTGGTGGCATTGGTTGCTGGAAAGAGAAA",
            "AATTGAGTCGTAGTATCCACACCAAGCCGGCGTTATCCGGTGAGGCGCAATGTTGCGGGGGCTTTATCCCTGGTGGCATTGGTTGCTGGAAAGAGAAA",
            "GTACAGCTCAGAAAGCGGAGTTGCGCCAAGATTGTTAACCAGCGCAATCACCCGATCGCCAGACTGGAGCGGTTGTTTGGTTTGTTGTTCTTCCTGCC",
            "TTACCTTGTGGAGCGACATCCAGAGGCACTTCACCGCTTGCCAGCGGCTTACCATCCAGCGCCACCATCCAGTGCAGGAGCTCGTTATCGCTATGACG"
        ])
        np.testing.assert_array_equal(expected_sequences, sequences)

    def test_import_fasta(self):
        sequences = self.importer.import_fasta(['../data/mini_read_for_test.fasta', '../data/mini_read_for_test_2.fasta.gz'])
        expected_sequences = np.array([
            "CTGTTGAGCTATACTGTGGGAAGTCTGCTTGGCCCGTCATTTACCGCTATGCTAATGCAGAATTTCTCCGATAATTTATTGTTTATCATGATCGCCAGCGTATCGTTTATCTATTTGCTG",
            "AATTTCTCCGATAATTTATTGTTTATCATGATCGCCAGCGTATCGTTTATCTATTTGCTGCTGTTGAGCTATACTGTGGGAAGTCTGCTTGGCCCGTCATTTACCGCTATGCTAATGCAG",
            "ATGATCGCCAGCGTATCGTTTATCTATTTGCTGAATTTCTCCGATAATTTATTGTTTATCGCTTGGCCCGTCATTTACCGCTATGCTAATGCAGCTGTTGAGCTATACTGTGGGAAGTCT",
            "ATGATCGCCAGCGTATCGTTTATCTATTTGCTGAATTTCTCCGATAATTTATTGTTTATCGCTTGGCCCGTCATTTACCGCTATGCTAATGCAGCTGTTGAGCTATACTGTGGGAAGTCT",
            "CTGTTGAGCTATACTGTGGGAAGTCTGCTTGGCCCGTCATTTACCGCTATGCTAATGCAGAATTTCTCCGATAATTTATTGTTTATCATGATCGCCAGCGTATCGTTTATCTATTTGCTG",
            "AATTTCTCCGATAATTTATTGTTTATCATGATCGCCAGCGTATCGTTTATCTATTTGCTGCTGTTGAGCTATACTGTGGGAAGTCTGCTTGGCCCGTCATTTACCGCTATGCTAATGCAG",
            "ATGATCGCCAGCGTATCGTTTATCTATTTGCTGAATTTCTCCGATAATTTATTGTTTATCGCTTGGCCCGTCATTTACCGCTATGCTAATGCAGCTGTTGAGCTATACTGTGGGAAGTCT"
        ])
        np.testing.assert_array_equal(expected_sequences, sequences)

    def test_import_fastq_with_reverse_complement(self):
        sequences = self.importer.import_fastq(['../data/mini_read_for_test.fastq', '../data/mini_read_for_test_2.fastq.gz'],
                                               True)
        expected_sequences = np.array([
            "AATTGAGTCGTAGTATCCACACCAAGCCGGCGTTATCCGGTGAGGCGCAATGTTGCGGGGGCTTTATCCCTGGTGGCATTGGTTGCTGGAAAGAGAAA",
            self.reverser.reverse_complement(
                "AATTGAGTCGTAGTATCCACACCAAGCCGGCGTTATCCGGTGAGGCGCAATGTTGCGGGGGCTTTATCCCTGGTGGCATTGGTTGCTGGAAAGAGAAA"),
            "AATTGAGTCGTAGTATCCACACCAAGCCGGCGTTATCCGGTGAGGCGCAATGTTGCGGGGGCTTTATCCCTGGTGGCATTGGTTGCTGGAAAGAGAAA",
            self.reverser.reverse_complement(
                "AATTGAGTCGTAGTATCCACACCAAGCCGGCGTTATCCGGTGAGGCGCAATGTTGCGGGGGCTTTATCCCTGGTGGCATTGGTTGCTGGAAAGAGAAA"),
            "GTACAGCTCAGAAAGCGGAGTTGCGCCAAGATTGTTAACCAGCGCAATCACCCGATCGCCAGACTGGAGCGGTTGTTTGGTTTGTTGTTCTTCCTGCC",
            self.reverser.reverse_complement(
                "GTACAGCTCAGAAAGCGGAGTTGCGCCAAGATTGTTAACCAGCGCAATCACCCGATCGCCAGACTGGAGCGGTTGTTTGGTTTGTTGTTCTTCCTGCC"),
            "TTACCTTGTGGAGCGACATCCAGAGGCACTTCACCGCTTGCCAGCGGCTTACCATCCAGCGCCACCATCCAGTGCAGGAGCTCGTTATCGCTATGACG",
            self.reverser.reverse_complement(
                "TTACCTTGTGGAGCGACATCCAGAGGCACTTCACCGCTTGCCAGCGGCTTACCATCCAGCGCCACCATCCAGTGCAGGAGCTCGTTATCGCTATGACG")
        ])
        np.testing.assert_array_equal(expected_sequences, sequences)

    def test_no_files_fastq(self):
        sequences = self.importer.import_fastq([], True)
        np.testing.assert_array_equal(sequences, [])

    def test_bogus_file_fastq(self):
        try:
            self.importer.import_fastq(['data/blahblahblah.fastq'])
            self.fail()
        except FileNotFoundError as e:
            pass
        except Exception as e:
            self.fail()

    def test_no_files_fasta(self):
        sequences = self.importer.import_fasta([])
        np.testing.assert_array_equal(sequences, [])

    def test_bogus_file_fasta(self):
        try:
            self.importer.import_fasta(['data/blahblahblah.fasta'])
            self.fail()
        except FileNotFoundError as e:
            pass
        except Exception as e:
            self.fail()