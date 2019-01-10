from unittest import TestCase

import numpy as np

from models.ParsedFastaRecord import ParsedFastaRecord
from models.ParsedFastqRecord import ParsedFastqRecord
from preprocess.SequenceParser import FormattedSequenceParser


class TestFormattedSequenceParser(TestCase):
    def setUp(self):
        self.parser = FormattedSequenceParser()

    def test_parse_fastq_newlines(self):
        # TODO: do we want to do validations on the FastQ?
        parsed_record = self.parser.parse_fastq("@SRR959238.1 HWI-ST897:104:C015GACXX:6:1101:1337:2122/1\n", "AATTGAGTCG\n", "+\n", "===+=AA=AD\n")
        self.assertEqual(parsed_record, ParsedFastqRecord("AATTGAGTCG", np.array([28, 28, 28, 10, 28, 32, 32, 28, 32, 35])))

    def test_parse_fastq_no_newlines(self):
        parsed_record = self.parser.parse_fastq("@SRR959238.1 HWI-ST897:104:C015GACXX:6:1101:1337:2122/1", "AATTGAGTCG", "+", "===+=AA=AD")
        self.assertEqual(parsed_record, ParsedFastqRecord("AATTGAGTCG", np.array([28, 28, 28, 10, 28, 32, 32, 28, 32, 35])))

    def test_parse_fasta_newlines(self):
        parsed_sequence = self.parser.parse_fasta([">gi|170079663|ref|NC_010473.1|:1000000-1001000",
                                                   "CTGTTGAGCTATACTGTGGGAAGTCTGCTTGGCCCGTCATTTACCGCTATGCTAATGCAG\n",
                                                   "AATTTCTCCGATAATTTATTGTTTATCATGATCGCCAGCGTATCGTTTATCTATTTGCTG\n"])
        self.assertEqual(parsed_sequence,
                         ParsedFastaRecord(">gi|170079663|ref|NC_010473.1|:1000000-1001000",
                                           "CTGTTGAGCTATACTGTGGGAAGTCTGCTTGGCCCGTCATTTACCGCTATGCTAATGCAGAATTTCTCCGATAATTTATTGTTTATCATGATCGCCAGCGTATCGTTTATCTATTTGCTG"))

    def test_parse_fasta_no_newlines(self):
        parsed_sequence = self.parser.parse_fasta([">gi|170079663|ref|NC_010473.1|:1000000-1001000",
                                                   "CTGTTGAGCTATACTGTGGGAAGTCTGCTTGGCCCGTCATTTACCGCTATGCTAATGCAG",
                                                   "AATTTCTCCGATAATTTATTGTTTATCATGATCGCCAGCGTATCGTTTATCTATTTGCTG"])
        self.assertEqual(parsed_sequence,
                         ParsedFastaRecord(">gi|170079663|ref|NC_010473.1|:1000000-1001000",
                                           "CTGTTGAGCTATACTGTGGGAAGTCTGCTTGGCCCGTCATTTACCGCTATGCTAATGCAGAATTTCTCCGATAATTTATTGTTTATCATGATCGCCAGCGTATCGTTTATCTATTTGCTG"))