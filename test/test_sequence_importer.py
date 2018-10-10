from unittest import TestCase

import numpy as np

from BaseQualityConverter import BaseQualityConverter
from SequenceImporter import SequenceImporter
from SequenceReverser import SequenceReverser


class TestSequenceImporter(TestCase):
    def setUp(self):
        self.importer = SequenceImporter()
        self.converter = BaseQualityConverter()
        self.reverser = SequenceReverser()

    def test_import_fastq(self):
        sequences = self.importer.import_fastq('data/mini_read_for_test.fastq')
        self.assertEqual(len(sequences), 3)
        self.assertEqual(sequences[0].sequence,
                         "AATTGAGTCGTAGTATCCACACCAAGCCGGCGTTATCCGGTGAGGCGCAATGTTGCGGGGGCTTTATCCCTGGTGGCATTGGTTGCTGGAAAGAGAAA")
        self.assertEqual(sequences[1].sequence,
                         "GTACAGCTCAGAAAGCGGAGTTGCGCCAAGATTGTTAACCAGCGCAATCACCCGATCGCCAGACTGGAGCGGTTGTTTGGTTTGTTGTTCTTCCTGCC")
        self.assertEqual(sequences[2].sequence,
                         "TTACCTTGTGGAGCGACATCCAGAGGCACTTCACCGCTTGCCAGCGGCTTACCATCCAGCGCCACCATCCAGTGCAGGAGCTCGTTATCGCTATGACG")
        np.testing.assert_array_equal(sequences[0].phred_quality, self.converter.convert_quality_to_phred(
            "===+=AA=AD<AA+<,AAEEDFEDFFFD8?DD?6;=BBDA;A;(?;999?'--5>A????&0;<>++(+44:>+3889>44>:2<?>A324>(33<AA"))
        np.testing.assert_array_equal(sequences[1].phred_quality, self.converter.convert_quality_to_phred(
            "CCCFFFFFHHHHHJJIJJIIHIJJJJJJJJIIJJJJJJJIJJJJJJJJHHHHHFFDDDDDDDDDDDCCDDDDDDDDDDDDDDDDDDDDDDEDDDDDCD"))
        np.testing.assert_array_equal(sequences[2].phred_quality, self.converter.convert_quality_to_phred(
            "CCCFFFFFHHHHHJJJJJJJJJJJJJJGIJJGJJJIJJJJJJJJJJJJHFFFFFFEEDEEDDDDDDDDDDDDDDDCDDDDDDDDDDDDDDDDDDDDDD"))

    def test_import_fastq_with_reverse_complement(self):
        sequences = self.importer.import_fastq('data/mini_read_for_test.fastq', True)
        self.assertEqual(len(sequences), 6)
        self.assertEqual(sequences[0].sequence,
                         "AATTGAGTCGTAGTATCCACACCAAGCCGGCGTTATCCGGTGAGGCGCAATGTTGCGGGGGCTTTATCCCTGGTGGCATTGGTTGCTGGAAAGAGAAA")
        self.assertEqual(sequences[1].sequence,
                         self.reverser.reverse_complement("AATTGAGTCGTAGTATCCACACCAAGCCGGCGTTATCCGGTGAGGCGCAATGTTGCGGGGGCTTTATCCCTGGTGGCATTGGTTGCTGGAAAGAGAAA"))
        self.assertEqual(sequences[2].sequence,
                         "GTACAGCTCAGAAAGCGGAGTTGCGCCAAGATTGTTAACCAGCGCAATCACCCGATCGCCAGACTGGAGCGGTTGTTTGGTTTGTTGTTCTTCCTGCC")
        self.assertEqual(sequences[3].sequence,
                         self.reverser.reverse_complement("GTACAGCTCAGAAAGCGGAGTTGCGCCAAGATTGTTAACCAGCGCAATCACCCGATCGCCAGACTGGAGCGGTTGTTTGGTTTGTTGTTCTTCCTGCC"))
        self.assertEqual(sequences[4].sequence,
                         "TTACCTTGTGGAGCGACATCCAGAGGCACTTCACCGCTTGCCAGCGGCTTACCATCCAGCGCCACCATCCAGTGCAGGAGCTCGTTATCGCTATGACG")
        self.assertEqual(sequences[5].sequence,
                         self.reverser.reverse_complement("TTACCTTGTGGAGCGACATCCAGAGGCACTTCACCGCTTGCCAGCGGCTTACCATCCAGCGCCACCATCCAGTGCAGGAGCTCGTTATCGCTATGACG"))
        np.testing.assert_array_equal(sequences[0].phred_quality, self.converter.convert_quality_to_phred(
            "===+=AA=AD<AA+<,AAEEDFEDFFFD8?DD?6;=BBDA;A;(?;999?'--5>A????&0;<>++(+44:>+3889>44>:2<?>A324>(33<AA"))
        np.testing.assert_array_equal(sequences[1].phred_quality, self.converter.convert_quality_to_phred(
            "===+=AA=AD<AA+<,AAEEDFEDFFFD8?DD?6;=BBDA;A;(?;999?'--5>A????&0;<>++(+44:>+3889>44>:2<?>A324>(33<AA"[::-1]))
        np.testing.assert_array_equal(sequences[2].phred_quality, self.converter.convert_quality_to_phred(
            "CCCFFFFFHHHHHJJIJJIIHIJJJJJJJJIIJJJJJJJIJJJJJJJJHHHHHFFDDDDDDDDDDDCCDDDDDDDDDDDDDDDDDDDDDDEDDDDDCD"))
        np.testing.assert_array_equal(sequences[3].phred_quality, self.converter.convert_quality_to_phred(
            "CCCFFFFFHHHHHJJIJJIIHIJJJJJJJJIIJJJJJJJIJJJJJJJJHHHHHFFDDDDDDDDDDDCCDDDDDDDDDDDDDDDDDDDDDDEDDDDDCD"[::-1]))
        np.testing.assert_array_equal(sequences[4].phred_quality, self.converter.convert_quality_to_phred(
            "CCCFFFFFHHHHHJJJJJJJJJJJJJJGIJJGJJJIJJJJJJJJJJJJHFFFFFFEEDEEDDDDDDDDDDDDDDDCDDDDDDDDDDDDDDDDDDDDDD"))
        np.testing.assert_array_equal(sequences[5].phred_quality, self.converter.convert_quality_to_phred(
            "CCCFFFFFHHHHHJJJJJJJJJJJJJJGIJJGJJJIJJJJJJJJJJJJHFFFFFFEEDEEDDDDDDDDDDDDDDDCDDDDDDDDDDDDDDDDDDDDDD"[::-1]))

    def test_import_fastq_gz(self):
        sequences = self.importer.import_fastq('data/mini_read_for_test.fastq.gz')
        self.assertEqual(len(sequences), 3)
        self.assertEqual(sequences[0].sequence,
                         "AATTGAGTCGTAGTATCCACACCAAGCCGGCGTTATCCGGTGAGGCGCAATGTTGCGGGGGCTTTATCCCTGGTGGCATTGGTTGCTGGAAAGAGAAA")
        self.assertEqual(sequences[1].sequence,
                         "GTACAGCTCAGAAAGCGGAGTTGCGCCAAGATTGTTAACCAGCGCAATCACCCGATCGCCAGACTGGAGCGGTTGTTTGGTTTGTTGTTCTTCCTGCC")
        self.assertEqual(sequences[2].sequence,
                         "TTACCTTGTGGAGCGACATCCAGAGGCACTTCACCGCTTGCCAGCGGCTTACCATCCAGCGCCACCATCCAGTGCAGGAGCTCGTTATCGCTATGACG")
        np.testing.assert_array_equal(sequences[0].phred_quality, self.converter.convert_quality_to_phred(
            "===+=AA=AD<AA+<,AAEEDFEDFFFD8?DD?6;=BBDA;A;(?;999?'--5>A????&0;<>++(+44:>+3889>44>:2<?>A324>(33<AA"))
        np.testing.assert_array_equal(sequences[1].phred_quality, self.converter.convert_quality_to_phred(
            "CCCFFFFFHHHHHJJIJJIIHIJJJJJJJJIIJJJJJJJIJJJJJJJJHHHHHFFDDDDDDDDDDDCCDDDDDDDDDDDDDDDDDDDDDDEDDDDDCD"))
        np.testing.assert_array_equal(sequences[2].phred_quality, self.converter.convert_quality_to_phred(
            "CCCFFFFFHHHHHJJJJJJJJJJJJJJGIJJGJJJIJJJJJJJJJJJJHFFFFFFEEDEEDDDDDDDDDDDDDDDCDDDDDDDDDDDDDDDDDDDDDD"))

    def test_import_fastq_gz_with_reverse_complement(self):
        sequences = self.importer.import_fastq('data/mini_read_for_test.fastq.gz', True)
        self.assertEqual(len(sequences), 6)
        self.assertEqual(sequences[0].sequence,
                         "AATTGAGTCGTAGTATCCACACCAAGCCGGCGTTATCCGGTGAGGCGCAATGTTGCGGGGGCTTTATCCCTGGTGGCATTGGTTGCTGGAAAGAGAAA")
        self.assertEqual(sequences[1].sequence,
                         self.reverser.reverse_complement(
                             "AATTGAGTCGTAGTATCCACACCAAGCCGGCGTTATCCGGTGAGGCGCAATGTTGCGGGGGCTTTATCCCTGGTGGCATTGGTTGCTGGAAAGAGAAA"))
        self.assertEqual(sequences[2].sequence,
                         "GTACAGCTCAGAAAGCGGAGTTGCGCCAAGATTGTTAACCAGCGCAATCACCCGATCGCCAGACTGGAGCGGTTGTTTGGTTTGTTGTTCTTCCTGCC")
        self.assertEqual(sequences[3].sequence,
                         self.reverser.reverse_complement(
                             "GTACAGCTCAGAAAGCGGAGTTGCGCCAAGATTGTTAACCAGCGCAATCACCCGATCGCCAGACTGGAGCGGTTGTTTGGTTTGTTGTTCTTCCTGCC"))
        self.assertEqual(sequences[4].sequence,
                         "TTACCTTGTGGAGCGACATCCAGAGGCACTTCACCGCTTGCCAGCGGCTTACCATCCAGCGCCACCATCCAGTGCAGGAGCTCGTTATCGCTATGACG")
        self.assertEqual(sequences[5].sequence,
                         self.reverser.reverse_complement(
                             "TTACCTTGTGGAGCGACATCCAGAGGCACTTCACCGCTTGCCAGCGGCTTACCATCCAGCGCCACCATCCAGTGCAGGAGCTCGTTATCGCTATGACG"))
        np.testing.assert_array_equal(sequences[0].phred_quality, self.converter.convert_quality_to_phred(
            "===+=AA=AD<AA+<,AAEEDFEDFFFD8?DD?6;=BBDA;A;(?;999?'--5>A????&0;<>++(+44:>+3889>44>:2<?>A324>(33<AA"))
        np.testing.assert_array_equal(sequences[1].phred_quality, self.converter.convert_quality_to_phred(
            "===+=AA=AD<AA+<,AAEEDFEDFFFD8?DD?6;=BBDA;A;(?;999?'--5>A????&0;<>++(+44:>+3889>44>:2<?>A324>(33<AA"[::-1]))
        np.testing.assert_array_equal(sequences[2].phred_quality, self.converter.convert_quality_to_phred(
            "CCCFFFFFHHHHHJJIJJIIHIJJJJJJJJIIJJJJJJJIJJJJJJJJHHHHHFFDDDDDDDDDDDCCDDDDDDDDDDDDDDDDDDDDDDEDDDDDCD"))
        np.testing.assert_array_equal(sequences[3].phred_quality, self.converter.convert_quality_to_phred(
            "CCCFFFFFHHHHHJJIJJIIHIJJJJJJJJIIJJJJJJJIJJJJJJJJHHHHHFFDDDDDDDDDDDCCDDDDDDDDDDDDDDDDDDDDDDEDDDDDCD"[::-1]))
        np.testing.assert_array_equal(sequences[4].phred_quality, self.converter.convert_quality_to_phred(
            "CCCFFFFFHHHHHJJJJJJJJJJJJJJGIJJGJJJIJJJJJJJJJJJJHFFFFFFEEDEEDDDDDDDDDDDDDDDCDDDDDDDDDDDDDDDDDDDDDD"))
        np.testing.assert_array_equal(sequences[5].phred_quality, self.converter.convert_quality_to_phred(
            "CCCFFFFFHHHHHJJJJJJJJJJJJJJGIJJGJJJIJJJJJJJJJJJJHFFFFFFEEDEEDDDDDDDDDDDDDDDCDDDDDDDDDDDDDDDDDDDDDD"[::-1]))

    def test_bogus_file(self):
        try:
            self.importer.import_fastq('data/blahblahblah.fastq')
            self.fail()
        except FileNotFoundError as e:
            pass
        except Exception as e:
            self.fail()