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

    def test_extract_input_output_from_sequence_with_N(self):
        extractor = SlidingWindowExtractor(1, 1, 1)
        input_matrix, input_quality_matrix, output_matrix, shifted_output_matrix = \
            extractor.extract_input_output_from_sequence(self.erroneous_parsed_fastqs)

        self.assertEqual(len(input_matrix), 2)
        self.assertEqual(len(input_matrix[0]), 1)
        self.assertEqual(input_quality_matrix.shape, (2, 1))
        self.assertEqual(len(output_matrix), 2)
        self.assertEqual(len(output_matrix[0]), 2)
        self.assertEqual(len(shifted_output_matrix), 2)
        self.assertEqual(len(shifted_output_matrix[0]), 2)
        expected_input = np.array([
            ["A"],
            ["G"]
        ])
        expected_input_quality = np.array([
            [28],
            [28]
        ])
        expected_output = np.array([
            ["A", "T"],
            ["G", "C"]
        ])
        expected_shifted_output = np.array([
            ["!", "A"],
            ["!", "G"]
        ])
        np.testing.assert_array_equal(input_matrix, expected_input)
        np.testing.assert_array_equal(input_quality_matrix, expected_input_quality)
        np.testing.assert_array_equal(output_matrix, expected_output)
        np.testing.assert_array_equal(shifted_output_matrix, expected_shifted_output)

    def test_extract_input_output_from_sequence_zero_spacing(self):
        extractor = SlidingWindowExtractor(4, 0, 4)
        input_matrix, input_quality_matrix, output_matrix, shifted_output_matrix = \
            extractor.extract_input_output_from_sequence(self.parsed_fastqs)

        self.assertEqual(len(input_matrix), 3)
        self.assertEqual(len(input_matrix[0]), 4)
        self.assertEqual(input_quality_matrix.shape, (3, 4))
        self.assertEqual(len(output_matrix), 3)
        self.assertEqual(len(output_matrix[0]), 8)
        self.assertEqual(len(shifted_output_matrix), 3)
        self.assertEqual(len(shifted_output_matrix[0]), 8)
        expected_input = np.array([
            ["A", "A", "T", "T"],
            ["A", "T", "T", "G"],
            ["T", "T", "G", "A"]
        ])
        expected_input_quality = np.array([
            [28, 28, 28, 10],
            [28, 28, 10, 28],
            [28, 10, 28, 32]
        ])
        expected_output = np.array([
            ["A", "A", "T", "T", "G", "A", "G", "T"],
            ["A", "T", "T", "G", "A", "G", "T", "C"],
            ["T", "T", "G", "A", "G", "T", "C", "G"]
        ])
        expected_shifted_output = np.array([
            ["!", "A", "A", "T", "T", "G", "A", "G"],
            ["!", "A", "T", "T", "G", "A", "G", "T"],
            ["!", "T", "T", "G", "A", "G", "T", "C"]
        ])
        np.testing.assert_array_equal(input_matrix, expected_input)
        np.testing.assert_array_equal(input_quality_matrix, expected_input_quality)
        np.testing.assert_array_equal(output_matrix, expected_output)
        np.testing.assert_array_equal(shifted_output_matrix, expected_shifted_output)

    def test_extract_input_output_from_sequence_minimal_spacing(self):
        extractor = SlidingWindowExtractor(1, 1, 1)
        input_matrix, input_quality_matrix, output_matrix, shifted_output_matrix = \
            extractor.extract_input_output_from_sequence(self.parsed_fastqs)

        self.assertEqual(len(input_matrix), 8)
        self.assertEqual(len(input_matrix[0]), 1)
        self.assertEqual(input_quality_matrix.shape, (8, 1))
        self.assertEqual(len(output_matrix), 8)
        self.assertEqual(len(output_matrix[0]), 2)
        self.assertEqual(len(shifted_output_matrix), 8)
        self.assertEqual(len(shifted_output_matrix[0]), 2)
        expected_input = np.array([
            ["A"],
            ["A"],
            ["T"],
            ["T"],
            ["G"],
            ["A"],
            ["G"],
            ["T"]
        ])
        expected_input_quality = np.array([
            [28],
            [28],
            [28],
            [10],
            [28],
            [32],
            [32],
            [28]
        ])
        expected_output = np.array([
            ["A", "T"],
            ["A", "T"],
            ["T", "G"],
            ["T", "A"],
            ["G", "G"],
            ["A", "T"],
            ["G", "C"],
            ["T", "G"]
        ])
        expected_shifted_output = np.array([
            ["!", "A"],
            ["!", "A"],
            ["!", "T"],
            ["!", "T"],
            ["!", "G"],
            ["!", "A"],
            ["!", "G"],
            ["!", "T"]
        ])
        np.testing.assert_array_equal(input_matrix, expected_input)
        np.testing.assert_array_equal(input_quality_matrix, expected_input_quality)
        np.testing.assert_array_equal(output_matrix, expected_output)
        np.testing.assert_array_equal(shifted_output_matrix, expected_shifted_output)

    def test_extract_input_output_from_sequence(self):
        extractor = SlidingWindowExtractor(4, 3, 2)
        input_matrix, input_quality_matrix, output_matrix, shifted_output_matrix = \
            extractor.extract_input_output_from_sequence(self.parsed_fastqs)

        self.assertEqual(len(input_matrix), 2)
        self.assertEqual(len(input_matrix[0]), 4)
        self.assertEqual(input_quality_matrix.shape, (2, 4))
        self.assertEqual(len(output_matrix), 2)
        self.assertEqual(len(output_matrix[0]), 6)
        self.assertEqual(len(shifted_output_matrix), 2)
        self.assertEqual(len(shifted_output_matrix[0]), 6)
        expected_input = np.array([
            ["A", "A", "T", "T"],
            ["A", "T", "T", "G"]
        ])
        expected_input_quality = np.array([
            [28, 28, 28, 10],
            [28, 28, 10, 28]
        ])
        expected_output = np.array([
            ["A", "A", "T", "T", "T", "C"],
            ["A", "T", "T", "G", "C", "G"]
        ])
        expected_shifted_output = np.array([
            ["!", "A", "A", "T", "T", "T"],
            ["!", "A", "T", "T", "G", "C"]
        ])
        np.testing.assert_array_equal(input_matrix, expected_input)
        np.testing.assert_array_equal(input_quality_matrix, expected_input_quality)
        np.testing.assert_array_equal(output_matrix, expected_output)
        np.testing.assert_array_equal(shifted_output_matrix, expected_shifted_output)

    def test_extract_input_output_from_sequence_too_much_input(self):
        extractor = SlidingWindowExtractor(11, 1, 1)
        input_matrix, input_quality_matrix, output_matrix, shifted_output_matrix = \
            extractor.extract_input_output_from_sequence(self.parsed_fastqs)

        self.assertEqual(len(input_matrix), 0)
        self.assertEqual(len(input_quality_matrix), 0)
        self.assertEqual(len(output_matrix), 0)
        self.assertEqual(len(shifted_output_matrix), 0)

    def test_extract_input_output_from_sequence_too_much_spacing(self):
        extractor = SlidingWindowExtractor(1, 11, 1)
        input_matrix, input_quality_matrix, output_matrix, shifted_output_matrix = \
            extractor.extract_input_output_from_sequence(self.parsed_fastqs)

        self.assertEqual(len(input_matrix), 0)
        self.assertEqual(len(input_quality_matrix), 0)
        self.assertEqual(len(output_matrix), 0)
        self.assertEqual(len(shifted_output_matrix), 0)

    def test_extract_input_output_from_sequence_too_much_output(self):
        extractor = SlidingWindowExtractor(1, 1, 11)
        input_matrix, input_quality_matrix, output_matrix, shifted_output_matrix = \
            extractor.extract_input_output_from_sequence(self.parsed_fastqs)

        self.assertEqual(len(input_matrix), 0)
        self.assertEqual(len(input_quality_matrix), 0)
        self.assertEqual(len(output_matrix), 0)
        self.assertEqual(len(shifted_output_matrix), 0)
