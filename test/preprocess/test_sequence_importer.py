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
        self.assertEqual(len(sequences), 3)
        self.assertEqual(sequences[0], "AATTGAGTCGTAGTATCCACACCAAGCCGGCGTTATCCGGTGAGGCGCAATGTTGCGGGGGCTTTATCCCTGGTGGCATTGGTTGCTGGAAAGAGAAA")
        self.assertEqual(sequences[1], "GTACAGCTCAGAAAGCGGAGTTGCGCCAAGATTGTTAACCAGCGCAATCACCCGATCGCCAGACTGGAGCGGTTGTTTGGTTTGTTGTTCTTCCTGCC")
        self.assertEqual(sequences[2], "TTACCTTGTGGAGCGACATCCAGAGGCACTTCACCGCTTGCCAGCGGCTTACCATCCAGCGCCACCATCCAGTGCAGGAGCTCGTTATCGCTATGACG")

    def test_import_fasta(self):
        sequences = self.importer.import_fasta(['../data/mini_read_for_test.fasta', '../data/mini_read_for_test_2.fasta.gz'])
        self.assertEqual(len(sequences), 6)
        self.assertEqual(sequences[0],
                         "CTGTTGAGCTATACTGTGGGAAGTCTGCTTGGCCCGTCATTTACCGCTATGCTAATGCAGAATTTCTCCGATAATTTATTGTTTATCATGATCGCCAGCGTATCGTTTATCTATTTGCTG")
        self.assertEqual(sequences[1],
                         "AATTTCTCCGATAATTTATTGTTTATCATGATCGCCAGCGTATCGTTTATCTATTTGCTGCTGTTGAGCTATACTGTGGGAAGTCTGCTTGGCCCGTCATTTACCGCTATGCTAATGCAG")
        self.assertEqual(sequences[2],
                         "ATGATCGCCAGCGTATCGTTTATCTATTTGCTGAATTTCTCCGATAATTTATTGTTTATCGCTTGGCCCGTCATTTACCGCTATGCTAATGCAGCTGTTGAGCTATACTGTGGGAAGTCT")
        self.assertEqual(sequences[3],
                         "CTGTTGAGCTATACTGTGGGAAGTCTGCTTGGCCCGTCATTTACCGCTATGCTAATGCAGAATTTCTCCGATAATTTATTGTTTATCATGATCGCCAGCGTATCGTTTATCTATTTGCTG")
        self.assertEqual(sequences[4],
                         "AATTTCTCCGATAATTTATTGTTTATCATGATCGCCAGCGTATCGTTTATCTATTTGCTGCTGTTGAGCTATACTGTGGGAAGTCTGCTTGGCCCGTCATTTACCGCTATGCTAATGCAG")
        self.assertEqual(sequences[5],
                        "ATGATCGCCAGCGTATCGTTTATCTATTTGCTGAATTTCTCCGATAATTTATTGTTTATCGCTTGGCCCGTCATTTACCGCTATGCTAATGCAGCTGTTGAGCTATACTGTGGGAAGTCT")

    def test_import_fastq_with_reverse_complement(self):
        sequences = self.importer.import_fastq(['../data/mini_read_for_test.fastq', '../data/mini_read_for_test_2.fastq.gz'],
                                               True)
        self.assertEqual(len(sequences), 6)
        self.assertEqual(sequences[0], "AATTGAGTCGTAGTATCCACACCAAGCCGGCGTTATCCGGTGAGGCGCAATGTTGCGGGGGCTTTATCCCTGGTGGCATTGGTTGCTGGAAAGAGAAA")
        self.assertEqual(sequences[1], self.reverser.reverse_complement("AATTGAGTCGTAGTATCCACACCAAGCCGGCGTTATCCGGTGAGGCGCAATGTTGCGGGGGCTTTATCCCTGGTGGCATTGGTTGCTGGAAAGAGAAA"))
        self.assertEqual(sequences[2], "GTACAGCTCAGAAAGCGGAGTTGCGCCAAGATTGTTAACCAGCGCAATCACCCGATCGCCAGACTGGAGCGGTTGTTTGGTTTGTTGTTCTTCCTGCC")
        self.assertEqual(sequences[3], self.reverser.reverse_complement("GTACAGCTCAGAAAGCGGAGTTGCGCCAAGATTGTTAACCAGCGCAATCACCCGATCGCCAGACTGGAGCGGTTGTTTGGTTTGTTGTTCTTCCTGCC"))
        self.assertEqual(sequences[4], "TTACCTTGTGGAGCGACATCCAGAGGCACTTCACCGCTTGCCAGCGGCTTACCATCCAGCGCCACCATCCAGTGCAGGAGCTCGTTATCGCTATGACG")
        self.assertEqual(sequences[5], self.reverser.reverse_complement("TTACCTTGTGGAGCGACATCCAGAGGCACTTCACCGCTTGCCAGCGGCTTACCATCCAGCGCCACCATCCAGTGCAGGAGCTCGTTATCGCTATGACG"))

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