import numpy as np
from unittest import TestCase

from SequenceParser import ParsedFastqRecord
from SlidingWindowExtractor import SlidingWindowExtractor, SlidingWindowParamException


class TestSlidingWindowExtractor(TestCase):
    def setUp(self):
        self.parsed_fastqs = [
            ParsedFastqRecord("AATTGAGTCG", np.array([28, 28, 28, 10, 28, 32, 32, 28, 32, 35]))
        ]

    def test_nonpositive_input_length(self):
        try:
            extractor = SlidingWindowExtractor(0, 1, 1)
        except SlidingWindowParamException:
            pass
        except:
            self.fail()

    def test_nonpositive_spacing_length(self):
        try:
            extractor = SlidingWindowExtractor(1, 0, 1)
        except SlidingWindowParamException:
            pass
        except:
            self.fail()

    def test_nonpositive_output_length(self):
        try:
            extractor = SlidingWindowExtractor(1, 1, 0)
        except SlidingWindowParamException:
            pass
        except:
            self.fail()

    def test_extract_input_output_from_sequence_minimal_spacing(self):
        extractor = SlidingWindowExtractor(1, 1, 1)
        input_output = extractor.extract_input_output_from_sequence(self.parsed_fastqs)
        input_matrix = input_output[0]
        input_quality_matrix = input_output[1]
        output_matrix = input_output[2]
        output_quality_matrix = input_output[3]

        self.assertEqual(len(input_matrix), 8)
        self.assertEqual(len(input_matrix[0]), 1)
        self.assertEqual(input_quality_matrix.shape, (8, 1))
        self.assertEqual(len(output_matrix), 8)
        self.assertEqual(len(output_matrix[0]), 1)
        self.assertEqual(output_quality_matrix.shape, (8, 1))
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
            ["T"],
            ["T"],
            ["G"],
            ["A"],
            ["G"],
            ["T"],
            ["C"],
            ["G"]
        ]
        expected_output_quality = np.array([
            [28],
            [10],
            [28],
            [32],
            [32],
            [28],
            [32],
            [35]
        ])
        np.testing.assert_array_equal(input_matrix, expected_input)
        np.testing.assert_array_equal(input_quality_matrix, expected_input_quality)
        np.testing.assert_array_equal(output_matrix, expected_output)
        np.testing.assert_array_equal(output_quality_matrix, expected_output_quality)

    def test_extract_input_output_from_sequence(self):
        extractor = SlidingWindowExtractor(4, 3, 2)
        input_output = extractor.extract_input_output_from_sequence(self.parsed_fastqs)
        input_matrix = input_output[0]
        input_quality_matrix = input_output[1]
        output_matrix = input_output[2]
        output_quality_matrix = input_output[3]

        self.assertEqual(len(input_matrix), 2)
        self.assertEqual(len(input_matrix[0]), 4)
        self.assertEqual(input_quality_matrix.shape, (2, 4))
        self.assertEqual(len(output_matrix), 2)
        self.assertEqual(len(output_matrix[0]), 2)
        self.assertEqual(output_quality_matrix.shape, (2, 2))
        expected_input = [
            ["A", "A", "T", "T"],
            ["A", "T", "T", "G"]
        ]
        expected_input_quality = np.array([
            [28, 28, 28, 10],
            [28, 28, 10, 28]
        ])
        expected_output = [
            ["T", "C"],
            ["C", "G"]
        ]
        expected_output_quality = np.array([
            [28, 32],
            [32, 35]
        ])
        np.testing.assert_array_equal(input_matrix, expected_input)
        np.testing.assert_array_equal(input_quality_matrix, expected_input_quality)
        np.testing.assert_array_equal(output_matrix, expected_output)
        np.testing.assert_array_equal(output_quality_matrix, expected_output_quality)

    def test_extract_input_output_from_sequence_too_much_input(self):
        extractor = SlidingWindowExtractor(11, 1, 1)
        input_output = extractor.extract_input_output_from_sequence(self.parsed_fastqs)
        input_matrix = input_output[0]
        input_quality_matrix = input_output[1]
        output_matrix = input_output[2]
        output_quality_matrix = input_output[3]

        self.assertEqual(len(input_matrix), 0)
        self.assertEqual(len(input_quality_matrix), 0)
        self.assertEqual(len(output_matrix), 0)
        self.assertEqual(len(output_quality_matrix), 0)

    def test_extract_input_output_from_sequence_too_much_spacing(self):
        extractor = SlidingWindowExtractor(1, 11, 1)
        input_output = extractor.extract_input_output_from_sequence(self.parsed_fastqs)
        input_matrix = input_output[0]
        input_quality_matrix = input_output[1]
        output_matrix = input_output[2]
        output_quality_matrix = input_output[3]

        self.assertEqual(len(input_matrix), 0)
        self.assertEqual(len(input_quality_matrix), 0)
        self.assertEqual(len(output_matrix), 0)
        self.assertEqual(len(output_quality_matrix), 0)

    def test_extract_input_output_from_sequence_too_much_output(self):
        extractor = SlidingWindowExtractor(1, 1, 11)
        input_output = extractor.extract_input_output_from_sequence(self.parsed_fastqs)
        input_matrix = input_output[0]
        input_quality_matrix = input_output[1]
        output_matrix = input_output[2]
        output_quality_matrix = input_output[3]

        self.assertEqual(len(input_matrix), 0)
        self.assertEqual(len(input_quality_matrix), 0)
        self.assertEqual(len(output_matrix), 0)
        self.assertEqual(len(output_quality_matrix), 0)
