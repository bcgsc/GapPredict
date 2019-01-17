from unittest import TestCase

import numpy as np

from preprocess.SlidingWindowExtractor import SlidingWindowExtractor, SlidingWindowParamException


class TestSlidingWindowExtractor(TestCase):
    def setUp(self):
        self.parsed_fastqs = np.array([
            "AATTGAGTCG"
        ])
        self.erroneous_parsed_fastqs = np.array([
            "AANTGNCG"
        ])
        self.parsed_redundant_fastqs = np.array([
            "AAAAAGGGGGTT"
        ])
        self.erroneous_parsed_redundant_fastqs = np.array([
            "AAAANGGGGNTT"
        ])

    def test_nonpositive_input_length(self):
        try:
            extractor = SlidingWindowExtractor(0, 1, 1)
            self.fail()
        except SlidingWindowParamException as e:
            pass
        except Exception as e:
            self.fail()

    def test_negative_spacing_length(self):
        try:
            extractor = SlidingWindowExtractor(1, -1, 1)
            self.fail()
        except SlidingWindowParamException as e:
            pass
        except Exception as e:
            self.fail()

    def test_nonpositive_output_length(self):
        try:
            extractor = SlidingWindowExtractor(1, 1, 0)
            self.fail()
        except SlidingWindowParamException as e:
            pass
        except Exception as e:
            self.fail()

    def test_extract_unique_kmers_from_sequence_with_N(self):
        extractor = SlidingWindowExtractor(1, 0, 1)
        input_kmers, output_kmers = \
            extractor.extract_kmers_from_sequence(self.erroneous_parsed_redundant_fastqs, unique=True)

        expected_input = [
            "A",
            "G",
            "T"
        ]
        expected_output = [
            "A",
            "G",
            "T"
        ]
        np.testing.assert_array_equal(input_kmers, expected_input)
        np.testing.assert_array_equal(output_kmers, expected_output)

    def test_extract_unique_kmers_from_sequence_zero_spacing(self):
        extractor = SlidingWindowExtractor(2, 0, 2)
        input_kmers, output_kmers = \
            extractor.extract_kmers_from_sequence(self.parsed_redundant_fastqs, unique=True)

        expected_input = [
            "AA",
            "AA",
            "AA",
            "AG",
            "GG",
            "GG",
            "GG"
        ]
        expected_output = [
            "AA",
            "AG",
            "GG",
            "GG",
            "GG",
            "GT",
            "TT"
        ]
        np.testing.assert_array_equal(input_kmers, expected_input)
        np.testing.assert_array_equal(output_kmers, expected_output)

    def test_extract_kmers_from_sequence_with_N(self):
        extractor = SlidingWindowExtractor(1, 1, 1)
        input_kmers, output_kmers = \
            extractor.extract_kmers_from_sequence(self.erroneous_parsed_fastqs)

        expected_input = [
            "A",
            "G"
        ]
        expected_output = [
            "T",
            "C"
        ]
        np.testing.assert_array_equal(input_kmers, expected_input)
        np.testing.assert_array_equal(output_kmers, expected_output)

    def test_extract_kmers_from_sequence_zero_spacing(self):
        extractor = SlidingWindowExtractor(4, 0, 4)
        input_kmers, output_kmers = \
            extractor.extract_kmers_from_sequence(self.parsed_fastqs)

        expected_input = [
            "AATT",
            "ATTG",
            "TTGA"
        ]
        expected_output = [
            "GAGT",
            "AGTC",
            "GTCG"
        ]
        np.testing.assert_array_equal(input_kmers, expected_input)
        np.testing.assert_array_equal(output_kmers, expected_output)

    def test_extract_kmers_from_sequence_minimal_spacing(self):
        extractor = SlidingWindowExtractor(1, 1, 1)
        input_kmers, output_kmers = \
            extractor.extract_kmers_from_sequence(self.parsed_fastqs)

        expected_input = [
            "A",
            "A",
            "T",
            "T",
            "G",
            "A",
            "G",
            "T"
        ]
        expected_output = [
            "T",
            "T",
            "G",
            "A",
            "G",
            "T",
            "C",
            "G"
        ]
        np.testing.assert_array_equal(input_kmers, expected_input)
        np.testing.assert_array_equal(output_kmers, expected_output)

    def test_extract_kmers_from_sequence(self):
        extractor = SlidingWindowExtractor(4, 3, 2)
        input_kmers, output_kmers = \
            extractor.extract_kmers_from_sequence(self.parsed_fastqs)

        expected_input = [
            "AATT",
            "ATTG"
        ]
        expected_output = [
            "TC",
            "CG"
        ]
        np.testing.assert_array_equal(input_kmers, expected_input)
        np.testing.assert_array_equal(output_kmers, expected_output)

    def test_extract_kmers_from_sequence_too_much_input(self):
        extractor = SlidingWindowExtractor(11, 1, 1)
        input_kmers, output_kmers = \
            extractor.extract_kmers_from_sequence(self.parsed_fastqs)

        self.assertEqual(len(input_kmers), 0)
        self.assertEqual(len(output_kmers), 0)

    def test_extract_kmers_from_sequence_too_much_spacing(self):
        extractor = SlidingWindowExtractor(1, 11, 1)
        input_kmers, output_kmers = \
            extractor.extract_kmers_from_sequence(self.parsed_fastqs)

        self.assertEqual(len(input_kmers), 0)
        self.assertEqual(len(output_kmers), 0)

    def test_extract_kmers_from_sequence_too_much_output(self):
        extractor = SlidingWindowExtractor(1, 1, 11)
        input_kmers, output_kmers = \
            extractor.extract_kmers_from_sequence(self.parsed_fastqs)

        self.assertEqual(len(input_kmers), 0)
        self.assertEqual(len(output_kmers), 0)