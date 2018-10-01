import numpy as np
from unittest import TestCase

from SequenceParser import ParsedFastqRecord
from SlidingWindowExtractor import SlidingWindowExtractor, SlidingWindowParamException


class TestSlidingWindowExtractor(TestCase):
    def setUp(self):
        self.parsed_fastq = ParsedFastqRecord("AATTGAGTCG", np.array([28, 28, 28, 10, 28, 32, 32, 28, 32, 35]))

    def test_nonpositive_input_length(self):
        try:
            extractor = SlidingWindowExtractor(0, 1, 1)
        except SlidingWindowParamException:
            return
        except:
            self.fail()


    def test_nonpositive_spacing_length(self):
        try:
            extractor = SlidingWindowExtractor(1, 0, 1)
        except SlidingWindowParamException:
            return
        except:
            self.fail()

    def test_nonpositive_output_length(self):
        try:
            extractor = SlidingWindowExtractor(1, 1, 0)
        except SlidingWindowParamException:
            return
        except:
            self.fail()

    def test_extract_input_output_from_sequence_minimal_spacing(self):
        extractor = SlidingWindowExtractor(1, 1, 1)
        input_output = extractor.extract_input_output_from_sequence(self.parsed_fastq)
        input = input_output[0]
        output = input_output[1]
        self.assertEqual(len(input), 8)
        self.assertEqual(len(output), 8)
        expected_input = [
            ParsedFastqRecord("A", np.array([28])),
            ParsedFastqRecord("A", np.array([28])),
            ParsedFastqRecord("T", np.array([28])),
            ParsedFastqRecord("T", np.array([10])),
            ParsedFastqRecord("G", np.array([28])),
            ParsedFastqRecord("A", np.array([32])),
            ParsedFastqRecord("G", np.array([32])),
            ParsedFastqRecord("T", np.array([28]))
        ]
        expected_output = [
            ParsedFastqRecord("T", np.array([28])),
            ParsedFastqRecord("T", np.array([10])),
            ParsedFastqRecord("G", np.array([28])),
            ParsedFastqRecord("A", np.array([32])),
            ParsedFastqRecord("G", np.array([32])),
            ParsedFastqRecord("T", np.array([28])),
            ParsedFastqRecord("C", np.array([32])),
            ParsedFastqRecord("G", np.array([35]))
        ]
        self.assertEqual(expected_input, input)
        self.assertEqual(expected_output, output)

    def test_extract_input_output_from_sequence(self):
        extractor = SlidingWindowExtractor(4, 3, 2)
        input_output = extractor.extract_input_output_from_sequence(self.parsed_fastq)
        input = input_output[0]
        output = input_output[1]
        self.assertEqual(len(input), 2)
        self.assertEqual(len(output), 2)
        expected_input = [
            ParsedFastqRecord("AATT", np.array([28, 28, 28, 10])),
            ParsedFastqRecord("ATTG", np.array([28, 28, 10, 28]))
        ]
        expected_output = [
            ParsedFastqRecord("TC", np.array([28, 32])),
            ParsedFastqRecord("CG", np.array([32, 35]))
        ]
        self.assertEqual(expected_input, input)
        self.assertEqual(expected_output, output)

    def test_extract_input_output_from_sequence_too_much_input(self):
        extractor = SlidingWindowExtractor(11, 1, 1)
        input_output = extractor.extract_input_output_from_sequence(self.parsed_fastq)
        input = input_output[0]
        output = input_output[1]
        self.assertEqual(len(input), 0)
        self.assertEqual(len(output), 0)

    def test_extract_input_output_from_sequence_too_much_spacing(self):
        extractor = SlidingWindowExtractor(1, 11, 1)
        input_output = extractor.extract_input_output_from_sequence(self.parsed_fastq)
        input = input_output[0]
        output = input_output[1]
        self.assertEqual(len(input), 0)
        self.assertEqual(len(output), 0)

    def test_extract_input_output_from_sequence_too_much_output(self):
        extractor = SlidingWindowExtractor(1, 1, 11)
        input_output = extractor.extract_input_output_from_sequence(self.parsed_fastq)
        input = input_output[0]
        output = input_output[1]
        self.assertEqual(len(input), 0)
        self.assertEqual(len(output), 0)
