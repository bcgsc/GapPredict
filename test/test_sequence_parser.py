import numpy as np
from unittest import TestCase

from SequenceParser import SequenceParser


class TestSequenceParser(TestCase):
    def setUp(self):
        self.parser = SequenceParser()

    def test_parse_fastq_newlines(self):
        # TODO: do we want to do validations on the FastQ?
        parsed_record = self.parser.parse_fastq("@SRR959238.1 HWI-ST897:104:C015GACXX:6:1101:1337:2122/1\n", "AATTGAGTCG\n", "+\n", "===+=AA=AD\n")
        self.assertEqual(parsed_record.sequence, "AATTGAGTCG")
        np.testing.assert_array_equal(parsed_record.phred_quality, np.array([28, 28, 28, 10, 28, 32, 32, 28, 32, 35]))

    def test_parse_fastq_no_newlines(self):
        parsed_record = self.parser.parse_fastq("@SRR959238.1 HWI-ST897:104:C015GACXX:6:1101:1337:2122/1", "AATTGAGTCG", "+", "===+=AA=AD")
        self.assertEqual(parsed_record.sequence, "AATTGAGTCG")
        np.testing.assert_array_equal(parsed_record.phred_quality, np.array([28, 28, 28, 10, 28, 32, 32, 28, 32, 35]))
