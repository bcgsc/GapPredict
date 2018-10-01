import numpy as np
from unittest import TestCase

from BaseQualityConverter import BaseQualityConverter
from SequenceImporter import SequenceImporter


class TestSequenceImporter(TestCase):
    def setUp(self):
        self.importer = SequenceImporter()
        self.converter = BaseQualityConverter()

    def test_import_fastq(self):
        sequences = self.importer.import_fastq('data/mini_read_for_test.fastq')
        self.assertEqual(len(sequences), 3)
        self.assertEqual(sequences[0].sequence, "AATTGAGTCGTAGTATCCACACCAAGCCGGCGTTATCCGGTGAGGCGCAATGTTGCGGGGGCTTTATCCCTGGTGGCATTGGTTGCTGGAAAGAGAAA")
        self.assertEqual(sequences[1].sequence, "GTACAGCTCAGAAAGCGGAGTTGCGCCAAGATTGTTAACCAGCGCAATCACCCGATCGCCAGACTGGAGCGGTTGTTTGGTTTGTTGTTCTTCCTGCC")
        self.assertEqual(sequences[2].sequence, "TTACCTTGTGGAGCGACATCCAGAGGCACTTCACCGCTTGCCAGCGGCTTACCATCCAGCGCCACCATCCAGTGCAGGAGCTCGTTATCGCTATGACG")
        np.testing.assert_array_equal(sequences[0].phred_quality, self.converter.convert_quality_to_phred("===+=AA=AD<AA+<,AAEEDFEDFFFD8?DD?6;=BBDA;A;(?;999?'--5>A????&0;<>++(+44:>+3889>44>:2<?>A324>(33<AA"))
        np.testing.assert_array_equal(sequences[1].phred_quality, self.converter.convert_quality_to_phred("CCCFFFFFHHHHHJJIJJIIHIJJJJJJJJIIJJJJJJJIJJJJJJJJHHHHHFFDDDDDDDDDDDCCDDDDDDDDDDDDDDDDDDDDDDEDDDDDCD"))
        np.testing.assert_array_equal(sequences[2].phred_quality, self.converter.convert_quality_to_phred("CCCFFFFFHHHHHJJJJJJJJJJJJJJGIJJGJJJIJJJJJJJJJJJJHFFFFFFEEDEEDDDDDDDDDDDDDDDCDDDDDDDDDDDDDDDDDDDDDD"))
        return

    def test_import_fastq_gz(self):
        sequences = self.importer.import_fastq('data/mini_read_for_test.fastq.gz')
        self.assertEqual(len(sequences), 3)
        self.assertEqual(sequences[0].sequence, "AATTGAGTCGTAGTATCCACACCAAGCCGGCGTTATCCGGTGAGGCGCAATGTTGCGGGGGCTTTATCCCTGGTGGCATTGGTTGCTGGAAAGAGAAA")
        self.assertEqual(sequences[1].sequence, "GTACAGCTCAGAAAGCGGAGTTGCGCCAAGATTGTTAACCAGCGCAATCACCCGATCGCCAGACTGGAGCGGTTGTTTGGTTTGTTGTTCTTCCTGCC")
        self.assertEqual(sequences[2].sequence, "TTACCTTGTGGAGCGACATCCAGAGGCACTTCACCGCTTGCCAGCGGCTTACCATCCAGCGCCACCATCCAGTGCAGGAGCTCGTTATCGCTATGACG")
        np.testing.assert_array_equal(sequences[0].phred_quality, self.converter.convert_quality_to_phred("===+=AA=AD<AA+<,AAEEDFEDFFFD8?DD?6;=BBDA;A;(?;999?'--5>A????&0;<>++(+44:>+3889>44>:2<?>A324>(33<AA"))
        np.testing.assert_array_equal(sequences[1].phred_quality, self.converter.convert_quality_to_phred("CCCFFFFFHHHHHJJIJJIIHIJJJJJJJJIIJJJJJJJIJJJJJJJJHHHHHFFDDDDDDDDDDDCCDDDDDDDDDDDDDDDDDDDDDDEDDDDDCD"))
        np.testing.assert_array_equal(sequences[2].phred_quality, self.converter.convert_quality_to_phred("CCCFFFFFHHHHHJJJJJJJJJJJJJJGIJJGJJJIJJJJJJJJJJJJHFFFFFFEEDEEDDDDDDDDDDDDDDDCDDDDDDDDDDDDDDDDDDDDDD"))

    def test_bogus_file(self):
        try:
            self.importer.import_fastq('data/blahblahblah.fastq')
        except FileNotFoundError as e:
            return
        except Exception:
            self.fail()