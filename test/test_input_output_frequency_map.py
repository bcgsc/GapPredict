from unittest import TestCase

from stats.InputOutputFrequencyMap import InputOutputFrequencyMap


class TestInputOutputFrequencyMap(TestCase):
    def setUp(self):
        self.map = InputOutputFrequencyMap()

    def test_load_input_outputs(self):
        inputs = [
            "A",
            "A",
            "T",
            "T",
            "G",
            "A",
            "G",
            "T"
        ]
        outputs = [
            "T",
            "T",
            "G",
            "A",
            "G",
            "T",
            "C",
            "G"
        ]
        self.map.load_input_outputs(inputs, outputs)
        map = self.map.get_map()
        expected_map = {
            "A": {
                "T": 3
            },
            "T": {
                "G": 2,
                "A": 1
            },
            "G": {
                "G": 1,
                "C": 1
            }
        }
        self.assertEqual(map, expected_map)

    def test_get_unique_mappings_per_input(self):
        inputs = [
            "A",
            "A",
            "T",
            "T",
            "G",
            "A",
            "G",
            "T"
        ]
        outputs = [
            "T",
            "T",
            "G",
            "A",
            "G",
            "T",
            "C",
            "G"
        ]
        self.map.load_input_outputs(inputs, outputs)
        mapping_map = self.map.get_unique_mappings_per_input()
        expected_map = {
            "A": 1,
            "T": 2,
            "G": 2
        }
        self.assertEqual(mapping_map, expected_map)

    def test_get_inputs_with_redundant_mappings(self):
        inputs = [
            "A",
            "A",
            "T",
            "T",
            "G",
            "A",
            "G",
            "T"
        ]
        outputs = [
            "T",
            "T",
            "G",
            "A",
            "G",
            "T",
            "C",
            "G"
        ]
        self.map.load_input_outputs(inputs, outputs)
        redundants = self.map.get_inputs_with_redundant_mappings()
        expected_redundants = {"T", "G"}
        self.assertEqual(redundants, expected_redundants)