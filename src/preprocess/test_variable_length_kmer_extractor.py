from unittest import TestCase

import numpy as np

from exceptions.SlidingWindowParamException import SlidingWindowParamException
from preprocess.VariableLengthKmerExtractor import VariableLengthKmerExtractor


class TestVariableLengthKmerExtractor(TestCase):
    def setUp(self):
        self.parsed_fastqs = [
            "AATTGAG",
            "ATGC"
        ]
        self.erroneous_parsed_fastqs = [
            "ACTAGNG",
            "NTGC"
        ]

    def test_nonpositive_input_length(self):
        try:
            extractor = VariableLengthKmerExtractor(0, 1, 1)
            self.fail()
        except SlidingWindowParamException as e:
            pass
        except Exception as e:
            self.fail()

    def test_negative_spacing_length(self):
        try:
            extractor = VariableLengthKmerExtractor(1, -1, 1)
            self.fail()
        except SlidingWindowParamException as e:
            pass
        except Exception as e:
            self.fail()

    def test_nonpositive_output_length(self):
        try:
            extractor = VariableLengthKmerExtractor(1, 1, 0)
            self.fail()
        except SlidingWindowParamException as e:
            pass
        except Exception as e:
            self.fail()

    def test_extract_kmers_from_sequence_minimal_spacing(self):
        extractor = VariableLengthKmerExtractor(1, 1, 1)
        input_kmers, output_kmers, max_length = \
            extractor.extract_kmers_from_sequence(self.parsed_fastqs)

        expected_input = [
            "A",
            "A",
            "T",
            "T",
            "G",
            "A",
            "T",
            "AA",
            "AT",
            "TT",
            "TG",
            "AT",
            "AAT",
            "ATT",
            "TTG",
            "AATT",
            "ATTG",
            "AATTG"
        ]
        expected_output = [
            "T",
            "T",
            "G",
            "A",
            "G",
            "G",
            "C",
            "T",
            "G",
            "A",
            "G",
            "C",
            "G",
            "A",
            "G",
            "A",
            "G",
            "G"
        ]
        np.testing.assert_array_equal(input_kmers, expected_input)
        np.testing.assert_array_equal(output_kmers, expected_output)
        self.assertEqual(max_length, 7)

    def test_extract_kmers_from_sequence_zero_spacing(self):
        extractor = VariableLengthKmerExtractor(2, 0, 2)
        input_kmers, output_kmers, max_length = \
            extractor.extract_kmers_from_sequence(self.parsed_fastqs)

        expected_input = [
            "AA",
            "AT",
            "TT",
            "TG",
            "AT",
            "AAT",
            "ATT",
            "TTG",
            "AATT",
            "ATTG",
            "AATTG"
        ]
        expected_output = [
            "TT",
            "TG",
            "GA",
            "AG",
            "GC",
            "TG",
            "GA",
            "AG",
            "GA",
            "AG",
            "AG"
        ]
        np.testing.assert_array_equal(input_kmers, expected_input)
        np.testing.assert_array_equal(output_kmers, expected_output)
        self.assertEqual(max_length, 7)

    def test_extract_kmers_from_sequence(self):
        extractor = VariableLengthKmerExtractor(3, 2, 1)
        input_kmers, output_kmers, max_length = \
            extractor.extract_kmers_from_sequence(self.parsed_fastqs)

        expected_input = [
            "AAT",
            "ATT",
            "AATT"
        ]
        expected_output = [
            "A",
            "G",
            "G"
        ]
        np.testing.assert_array_equal(input_kmers, expected_input)
        np.testing.assert_array_equal(output_kmers, expected_output)
        self.assertEqual(max_length, 7)

    def test_extract_kmers_from_sequence_with_N(self):
        extractor = VariableLengthKmerExtractor(1, 0, 1)
        input_kmers, output_kmers, max_length = \
            extractor.extract_kmers_from_sequence(self.erroneous_parsed_fastqs)

        expected_input = [
            "A",
            "C",
            "T",
            "A",
            "T",
            "G",
            "AC",
            "CT",
            "TA",
            "TG",
            "ACT",
            "CTA",
            "ACTA",
        ]
        expected_output = [
            "C",
            "T",
            "A",
            "G",
            "G",
            "C",
            "T",
            "A",
            "G",
            "C",
            "A",
            "G",
            "G",
        ]
        np.testing.assert_array_equal(input_kmers, expected_input)
        np.testing.assert_array_equal(output_kmers, expected_output)
        self.assertEqual(max_length, 7)

    def test_extract_kmers_from_sequence_lower_bound_too_long(self):
        extractor = VariableLengthKmerExtractor(8, 1, 1)
        input_kmers, output_kmers, max_length = \
            extractor.extract_kmers_from_sequence(self.parsed_fastqs)

        self.assertEqual(len(input_kmers), 0)
        self.assertEqual(len(output_kmers), 0)
        self.assertEqual(max_length, 7)

    def test_extract_kmers_from_sequence_spacing_too_long(self):
        extractor = VariableLengthKmerExtractor(1, 8, 1)
        input_kmers, output_kmers, max_length = \
            extractor.extract_kmers_from_sequence(self.parsed_fastqs)

        self.assertEqual(len(input_kmers), 0)
        self.assertEqual(len(output_kmers), 0)
        self.assertEqual(max_length, 7)

    def test_extract_kmers_from_sequence_output_too_long(self):
        extractor = VariableLengthKmerExtractor(1, 1, 8)
        input_kmers, output_kmers, max_length = \
            extractor.extract_kmers_from_sequence(self.parsed_fastqs)

        self.assertEqual(len(input_kmers), 0)
        self.assertEqual(len(output_kmers), 0)
        self.assertEqual(max_length, 7)