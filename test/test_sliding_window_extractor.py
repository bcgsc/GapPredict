from unittest import TestCase

import numpy as np

from SlidingWindowExtractor import SlidingWindowExtractor, SlidingWindowParamException
from models.ParsedFastqRecord import ParsedFastqRecord


class TestSlidingWindowExtractor(TestCase):
    def setUp(self):
        self.parsed_fastqs = [
            ParsedFastqRecord("AATTGAGTCG", np.array([28, 28, 28, 10, 28, 32, 32, 28, 32, 35]))
        ]
        self.erroneous_parsed_fastqs = [
            ParsedFastqRecord("AANTGNCG", np.array([28, 28, 28, 10, 28, 10, 13, 15]))
        ]

    def test_nonpositive_input_length(self):
        try:
            extractor = SlidingWindowExtractor(0, 1, 1)
            self.fail()
        except SlidingWindowParamException as e:
            pass
        except Exception as e:
            self.fail()

    def test_nonpositive_spacing_length(self):
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

    def test_kmers_from_sequence_with_N(self):
        extractor = SlidingWindowExtractor(1, 1, 1)
        input_kmers, output_kmers, quality_vectors = \
            extractor.extract_kmers_from_sequence(self.erroneous_parsed_fastqs)

        self.assertEqual(len(input_kmers), 2)
        self.assertEqual(len(input_kmers[0]), 1)
        self.assertEqual(len(output_kmers), 2)
        self.assertEqual(len(output_kmers[0]), 1)
        self.assertEqual(len(quality_vectors), 2)
        self.assertEqual(len(quality_vectors[0]), 1)
        expected_input = [
            "A",
            "G"
        ]
        expected_input_quality = [
            [28],
            [28]
        ]
        expected_output = [
            "T",
            "C"
        ]
        np.testing.assert_array_equal(input_kmers, expected_input)
        np.testing.assert_array_equal(quality_vectors, expected_input_quality)
        np.testing.assert_array_equal(output_kmers, expected_output)

    def test_extract_kmers_from_sequence_zero_spacing(self):
        extractor = SlidingWindowExtractor(4, 0, 4)
        input_kmers, output_kmers, quality_vectors = \
            extractor.extract_kmers_from_sequence(self.parsed_fastqs)

        self.assertEqual(len(input_kmers), 3)
        self.assertEqual(len(input_kmers[0]), 4)
        self.assertEqual(len(output_kmers), 3)
        self.assertEqual(len(output_kmers[0]), 4)
        self.assertEqual(len(quality_vectors), 3)
        self.assertEqual(len(quality_vectors[0]), 4)
        expected_input = [
            "AATT",
            "ATTG",
            "TTGA"
        ]
        expected_input_quality = [
            [28, 28, 28, 10],
            [28, 28, 10, 28],
            [28, 10, 28, 32]
        ]
        expected_output = [
            "GAGT",
            "AGTC",
            "GTCG"
        ]
        np.testing.assert_array_equal(input_kmers, expected_input)
        np.testing.assert_array_equal(quality_vectors, expected_input_quality)
        np.testing.assert_array_equal(output_kmers, expected_output)

    def test_extract_kmers_from_sequence_minimal_spacing(self):
        extractor = SlidingWindowExtractor(1, 1, 1)
        input_kmers, output_kmers, quality_vectors = \
            extractor.extract_kmers_from_sequence(self.parsed_fastqs)

        self.assertEqual(len(input_kmers), 8)
        self.assertEqual(len(input_kmers[0]), 1)
        self.assertEqual(len(output_kmers), 8)
        self.assertEqual(len(output_kmers[0]), 1)
        self.assertEqual(len(quality_vectors), 8)
        self.assertEqual(len(quality_vectors[0]), 1)
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
        expected_input_quality = [
            [28],
            [28],
            [28],
            [10],
            [28],
            [32],
            [32],
            [28]
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
        np.testing.assert_array_equal(quality_vectors, expected_input_quality)
        np.testing.assert_array_equal(output_kmers, expected_output)

    def test_extract_kmers_from_sequence(self):
        extractor = SlidingWindowExtractor(4, 3, 2)
        input_kmers, output_kmers, quality_vectors = \
            extractor.extract_kmers_from_sequence(self.parsed_fastqs)

        self.assertEqual(len(input_kmers), 2)
        self.assertEqual(len(input_kmers[0]), 4)
        self.assertEqual(len(output_kmers), 2)
        self.assertEqual(len(output_kmers[0]), 2)
        self.assertEqual(len(quality_vectors), 2)
        self.assertEqual(len(quality_vectors[0]), 4)
        expected_input = [
            "AATT",
            "ATTG"
        ]
        expected_input_quality = [
            [28, 28, 28, 10],
            [28, 28, 10, 28]
        ]
        expected_output = [
            "TC",
            "CG"
        ]
        np.testing.assert_array_equal(input_kmers, expected_input)
        np.testing.assert_array_equal(quality_vectors, expected_input_quality)
        np.testing.assert_array_equal(output_kmers, expected_output)

    def test_extract_kmers_from_sequence_too_much_input(self):
        extractor = SlidingWindowExtractor(11, 1, 1)
        input_kmers, output_kmers, quality_vectors = \
            extractor.extract_kmers_from_sequence(self.parsed_fastqs)

        self.assertEqual(len(input_kmers), 0)
        self.assertEqual(len(output_kmers), 0)
        self.assertEqual(len(quality_vectors), 0)

    def test_extract_kmers_from_sequence_too_much_spacing(self):
        extractor = SlidingWindowExtractor(1, 11, 1)
        input_kmers, output_kmers, quality_vectors = \
            extractor.extract_kmers_from_sequence(self.parsed_fastqs)

        self.assertEqual(len(input_kmers), 0)
        self.assertEqual(len(output_kmers), 0)
        self.assertEqual(len(quality_vectors), 0)

    def test_extract_kmers_from_sequence_too_much_output(self):
        extractor = SlidingWindowExtractor(1, 1, 11)
        input_kmers, output_kmers, quality_vectors = \
            extractor.extract_kmers_from_sequence(self.parsed_fastqs)

        self.assertEqual(len(input_kmers), 0)
        self.assertEqual(len(output_kmers), 0)
        self.assertEqual(len(quality_vectors), 0)