from unittest import TestCase

import numpy as np

from models.ParsedFastaRecord import ParsedFastaRecord
from models.ParsedFastqRecord import ParsedFastqRecord
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
        self.assertEqual(sequences[0], ParsedFastqRecord("AATTGAGTCGTAGTATCCACACCAAGCCGGCGTTATCCGGTGAGGCGCAATGTTGCGGGGGCTTTATCCCTGGTGGCATTGGTTGCTGGAAAGAGAAA",
                                                         self.converter.convert_quality_to_phred("===+=AA=AD<AA+<,AAEEDFEDFFFD8?DD?6;=BBDA;A;(?;999?'--5>A????&0;<>++(+44:>+3889>44>:2<?>A324>(33<AA")))
        self.assertEqual(sequences[1], ParsedFastqRecord("GTACAGCTCAGAAAGCGGAGTTGCGCCAAGATTGTTAACCAGCGCAATCACCCGATCGCCAGACTGGAGCGGTTGTTTGGTTTGTTGTTCTTCCTGCC",
                                                         self.converter.convert_quality_to_phred("CCCFFFFFHHHHHJJIJJIIHIJJJJJJJJIIJJJJJJJIJJJJJJJJHHHHHFFDDDDDDDDDDDCCDDDDDDDDDDDDDDDDDDDDDDEDDDDDCD")))
        self.assertEqual(sequences[2], ParsedFastqRecord("TTACCTTGTGGAGCGACATCCAGAGGCACTTCACCGCTTGCCAGCGGCTTACCATCCAGCGCCACCATCCAGTGCAGGAGCTCGTTATCGCTATGACG",
                                                         self.converter.convert_quality_to_phred("CCCFFFFFHHHHHJJJJJJJJJJJJJJGIJJGJJJIJJJJJJJJJJJJHFFFFFFEEDEEDDDDDDDDDDDDDDDCDDDDDDDDDDDDDDDDDDDDDD")))

    def test_import_fasta(self):
        sequences = self.importer.import_fasta(['../data/mini_read_for_test.fasta', '../data/mini_read_for_test_2.fasta.gz'])
        self.assertEqual(len(sequences), 6)
        self.assertEqual(sequences[0],
                         ParsedFastaRecord(">gi|170079663|ref|NC_010473.1|:1000000-1001000", "CTGTTGAGCTATACTGTGGGAAGTCTGCTTGGCCCGTCATTTACCGCTATGCTAATGCAGAATTTCTCCGATAATTTATTGTTTATCATGATCGCCAGCGTATCGTTTATCTATTTGCTG"))
        self.assertEqual(sequences[1],
                         ParsedFastaRecord(">gi|170079663|ref|NC_010473.1|:1001000-1002000", "AATTTCTCCGATAATTTATTGTTTATCATGATCGCCAGCGTATCGTTTATCTATTTGCTGCTGTTGAGCTATACTGTGGGAAGTCTGCTTGGCCCGTCATTTACCGCTATGCTAATGCAG"))
        self.assertEqual(sequences[2],
                         ParsedFastaRecord(">gi|170079663|ref|NC_010473.1|:1002000-1003000", "ATGATCGCCAGCGTATCGTTTATCTATTTGCTGAATTTCTCCGATAATTTATTGTTTATCGCTTGGCCCGTCATTTACCGCTATGCTAATGCAGCTGTTGAGCTATACTGTGGGAAGTCT"))
        self.assertEqual(sequences[3],
                         ParsedFastaRecord(">gi|170079663|ref|NC_010473.1|:1000000-1001000", "CTGTTGAGCTATACTGTGGGAAGTCTGCTTGGCCCGTCATTTACCGCTATGCTAATGCAGAATTTCTCCGATAATTTATTGTTTATCATGATCGCCAGCGTATCGTTTATCTATTTGCTG"))
        self.assertEqual(sequences[4],
                         ParsedFastaRecord(">gi|170079663|ref|NC_010473.1|:1001000-1002000", "AATTTCTCCGATAATTTATTGTTTATCATGATCGCCAGCGTATCGTTTATCTATTTGCTGCTGTTGAGCTATACTGTGGGAAGTCTGCTTGGCCCGTCATTTACCGCTATGCTAATGCAG"))
        self.assertEqual(sequences[5],
                         ParsedFastaRecord(">gi|170079663|ref|NC_010473.1|:1002000-1003000", "ATGATCGCCAGCGTATCGTTTATCTATTTGCTGAATTTCTCCGATAATTTATTGTTTATCGCTTGGCCCGTCATTTACCGCTATGCTAATGCAGCTGTTGAGCTATACTGTGGGAAGTCT"))

    def test_import_fastq_with_reverse_complement(self):
        sequences = self.importer.import_fastq(['../data/mini_read_for_test.fastq', '../data/mini_read_for_test_2.fastq.gz'],
                                               True)
        self.assertEqual(len(sequences), 6)
        self.assertEqual(sequences[0], ParsedFastqRecord("AATTGAGTCGTAGTATCCACACCAAGCCGGCGTTATCCGGTGAGGCGCAATGTTGCGGGGGCTTTATCCCTGGTGGCATTGGTTGCTGGAAAGAGAAA",
                                                         self.converter.convert_quality_to_phred("===+=AA=AD<AA+<,AAEEDFEDFFFD8?DD?6;=BBDA;A;(?;999?'--5>A????&0;<>++(+44:>+3889>44>:2<?>A324>(33<AA")))
        self.assertEqual(sequences[1], ParsedFastqRecord(self.reverser.reverse_complement("AATTGAGTCGTAGTATCCACACCAAGCCGGCGTTATCCGGTGAGGCGCAATGTTGCGGGGGCTTTATCCCTGGTGGCATTGGTTGCTGGAAAGAGAAA"),
                                                         self.converter.convert_quality_to_phred("===+=AA=AD<AA+<,AAEEDFEDFFFD8?DD?6;=BBDA;A;(?;999?'--5>A????&0;<>++(+44:>+3889>44>:2<?>A324>(33<AA"[::-1])))
        self.assertEqual(sequences[2], ParsedFastqRecord("GTACAGCTCAGAAAGCGGAGTTGCGCCAAGATTGTTAACCAGCGCAATCACCCGATCGCCAGACTGGAGCGGTTGTTTGGTTTGTTGTTCTTCCTGCC",
                                                         self.converter.convert_quality_to_phred("CCCFFFFFHHHHHJJIJJIIHIJJJJJJJJIIJJJJJJJIJJJJJJJJHHHHHFFDDDDDDDDDDDCCDDDDDDDDDDDDDDDDDDDDDDEDDDDDCD")))
        self.assertEqual(sequences[3], ParsedFastqRecord(self.reverser.reverse_complement("GTACAGCTCAGAAAGCGGAGTTGCGCCAAGATTGTTAACCAGCGCAATCACCCGATCGCCAGACTGGAGCGGTTGTTTGGTTTGTTGTTCTTCCTGCC"),
                                                         self.converter.convert_quality_to_phred("CCCFFFFFHHHHHJJIJJIIHIJJJJJJJJIIJJJJJJJIJJJJJJJJHHHHHFFDDDDDDDDDDDCCDDDDDDDDDDDDDDDDDDDDDDEDDDDDCD")[::-1]))
        self.assertEqual(sequences[4], ParsedFastqRecord("TTACCTTGTGGAGCGACATCCAGAGGCACTTCACCGCTTGCCAGCGGCTTACCATCCAGCGCCACCATCCAGTGCAGGAGCTCGTTATCGCTATGACG",
                                                         self.converter.convert_quality_to_phred("CCCFFFFFHHHHHJJJJJJJJJJJJJJGIJJGJJJIJJJJJJJJJJJJHFFFFFFEEDEEDDDDDDDDDDDDDDDCDDDDDDDDDDDDDDDDDDDDDD")))
        self.assertEqual(sequences[5], ParsedFastqRecord(self.reverser.reverse_complement("TTACCTTGTGGAGCGACATCCAGAGGCACTTCACCGCTTGCCAGCGGCTTACCATCCAGCGCCACCATCCAGTGCAGGAGCTCGTTATCGCTATGACG"),
                                                         self.converter.convert_quality_to_phred("CCCFFFFFHHHHHJJJJJJJJJJJJJJGIJJGJJJIJJJJJJJJJJJJHFFFFFFEEDEEDDDDDDDDDDDDDDDCDDDDDDDDDDDDDDDDDDDDDD")[::-1]))

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