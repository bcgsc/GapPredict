import sys
sys.path.append('../../')

from unittest import TestCase

import numpy as np

from predict.new_rnn.DataGenerator import DataGenerator
from preprocess.KmerLabelEncoder import KmerLabelEncoder
from preprocess.SequenceImporter import SequenceImporter


class TestDataGenerator(TestCase):
    def setUp(self):
        self.reads = SequenceImporter().import_fastq(['data/sample.fastq'])
        self.error_reads = SequenceImporter().import_fastq(['data/sample_with_errors.fastq'])
        self.batch_size = 4
        self.generator = DataGenerator(self.reads, 26, batch_size=self.batch_size, log_samples=False)
        self.error_generator = DataGenerator(self.error_reads, 26, batch_size=self.batch_size, log_samples=False, spacing=10)
        self.encoder = KmerLabelEncoder()

    def test_process_batch(self):
        np.random.seed(0)
        total_reads = len(self.reads)
        idx = np.arange(total_reads)
        np.random.shuffle(idx)

        batch_idx = idx[0:self.batch_size]
        batch = self.reads[batch_idx]
        expected_batch = np.array([
            "GCCGCTGGCTTTACGCTATATCCGGTGGCGATGGCATGGGCTTGCGAGAAAGTTGAACATCATCAACTGGTGGC",
            "GTAAGGCCTGGTTCATCGCCACCAGTTGATGATGTTCAACTTTCTCGCAAGCCCATGCCATCGCCAC",
            "CCTCGGTGCCGCTGGCTTTACGCTATATCCGGTGGCGATGGCATGGGCTTGCGAGAAAGTTGAACATCATCAACTGGTGGCGATGAACCAGGCCTTAC",
            "TCATGGGTGTTTATCTTTGCACTCTATTTCACACCTTATGCGCTAATTGTTGGTCAGTTAGGCTCGACATTCAAAGATGGGAAGGGCGGGGTCAGTAC"

        ], dtype='U98')
        np.testing.assert_array_equal(batch, expected_batch)

        np.random.seed(0)
        input, output = self.generator.__getitem__(0)

        expected_processed_batch = np.array([
            'TATCCGGTGGCGATGGCATGGGCTTGCGAGAAAGTTGAACATCATCA',
            'GGCCTGGTTCATCGCCACCAGTTGATGATGTTCAACTTTCTCGCAAG',
            'TATATCCGGTGGCGATGGCATGGGCTTGCGAGAAAGTTGAACATCAT',
            'GTGTTTATCTTTGCACTCTATTTCACACCTTATGCGCTAATTGTTGG'
        ])
        X = self.encoder.encode_kmers(expected_processed_batch, [], with_shifted_output=False)[0]

        np.testing.assert_array_equal(input, X)
        np.testing.assert_array_equal(output, np.array([[1, 0, 0, 0],
                                                        [0, 1, 0, 0],
                                                        [0, 1, 0, 0],
                                                        [0, 0, 0, 1]]))

    def test_process_batch_with_errors(self):
        np.random.seed(0)
        total_reads = len(self.error_reads)
        idx = np.arange(total_reads)
        np.random.shuffle(idx)

        batch_idx = idx[0:self.batch_size]
        batch = self.error_reads[batch_idx]
        expected_batch = np.array([
            "GCCGCTGGCTTTACGCTATATCCGGTGGCGATGGNATGGGCTTGCGAGAAAGTTGAACATCATCAACTGGTGGC",
            "GTAAGGCCTGGTTCATCGCCACCAGTTGATGATGTTCAACTTTCTCGCAAGCCCATGCCATCGCNAC",
            "CCTNGGTGCCGCTGGCTTTACGCTATATCCGGTGGCGATGGCATGGGCTTGCGAGAAAGTTGAACATCATCAACTGGTGGCGATGAACCAGGCCTTAC",
            "TCATGGGTGTTTATCTTTGCACTCTATTTCACACCTTATGCGCTAATTGTTGGTCAGTTAGGCTCGACATTCAAANATGGGAAGGGCGGGGTCAGTAN"

        ], dtype='U98')
        np.testing.assert_array_equal(batch, expected_batch)

        np.random.seed(0)
        input, output = self.error_generator.__getitem__(0)

        expected_processed_batch = np.array([
            'TGCCGCTGGCTTTACGCTATATCCGGTGGCGATGGCATGGGCTTGCG',
            'TATTTCACACCTTATGCGCTAATTGTTGGTCAGTTAGGCTCGACATT'
        ])
        X = self.encoder.encode_kmers(expected_processed_batch, [], with_shifted_output=False)[0]

        np.testing.assert_array_equal(input, X)
        np.testing.assert_array_equal(output, np.array([[1, 0, 0, 0],
                                                        [1, 0, 0, 0]]))

