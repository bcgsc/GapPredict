from unittest import TestCase

import numpy as np

from predict.new_rnn.DataGenerator import DataGenerator
from preprocess.SequenceImporter import SequenceImporter
from preprocess.SequenceSplitEncoder import SequenceSplitEncoder


class TestDataGenerator(TestCase):
    def setUp(self):
        self.reads = SequenceImporter().import_fastq(['data/sample.fastq'])
        self.generator = DataGenerator(self.reads, 26, batch_size=2)

    def test__process_batch(self):
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
        X, y, shifted_y = splitter.split_sequences(expected_batch)

        np.testing.assert_array_equal(input[0], X)
        np.testing.assert_array_equal(input[1], shifted_y)
        np.testing.assert_array_equal(output, y)



