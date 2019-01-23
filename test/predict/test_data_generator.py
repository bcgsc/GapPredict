from unittest import TestCase

import numpy as np

from predict.new_rnn.DataGenerator import DataGenerator
from preprocess.SequenceImporter import SequenceImporter
from preprocess.SequenceSplitEncoder import SequenceSplitEncoder


class TestDataGenerator(TestCase):
    def setUp(self):
        self.reads = SequenceImporter().import_fastq(['data/sample.fastq'])
        self.error_reads = SequenceImporter().import_fastq(['data/sample_with_errors.fastq'])
        self.generator = DataGenerator(self.reads, 26, batch_size=2)
        self.error_generator = DataGenerator(self.error_reads, 26, batch_size=2)

    #TODO: fix
    def test_process_batch(self):
        np.random.seed(0)
        batch_idx = np.random.randint(len(self.reads), size=2)
        batch = self.reads[batch_idx]
        expected_batch = np.array([
            "GTAAGGCCTGGTTCATCGCCACCAGTTGATGATGTTCAACTTTCTCGCAAGCCCATGCCATCGCCACCGGATATAGCGTAAAGCCAGCGGCACCGAGA",
            "CCTCGGTGCCGCTGGCTTTACGCTATATCCGGTGGCGATGGCATGGGCTTGCGAGAAAGTTGAACATCATCAACTGGTGGCGATGAAC"
        ], dtype='U98')
        np.testing.assert_array_equal(batch, expected_batch)

        np.random.seed(0)
        input, output = self.generator.__getitem__(0)

        splitter = SequenceSplitEncoder(26)
        X, y = splitter.split_sequences(expected_batch)

        np.testing.assert_array_equal(input, X)
        np.testing.assert_array_equal(output, np.array([[0, 0, 0, 1],
                                                        [1, 0, 0, 0]]))

    def test_process_batch_with_errors(self):
        np.random.seed(0)
        batch_idx = np.random.randint(len(self.error_reads), size=2)
        batch = self.error_reads[batch_idx]
        expected_batch = np.array([
            "GTAAGGCCTGGTTCATCGCCACCAGNTGATGATGTTCAACTTTCTCGCAAGCCCATGCCATCGCCACCGGATATAGCGTAAAGCCAGCGGCACCGAGA",
            "CCTCGGTGCCGCTGGCTTTACGCTATNTCCGGTGGCGATGGCATGGGCTTGCGAGAAAGTTGAACATCATCAACTGGTGGCGATGAAC"
        ], dtype='U98')
        np.testing.assert_array_equal(batch, expected_batch)

        np.random.seed(0)
        input, output = self.error_generator.__getitem__(0)

        self.assertEqual(len(input), 0)
        self.assertEqual(len(output), 0)



