from unittest import TestCase
from test.predict.MockStatefulGapPredictModel import MockStatefulGapPredictModel
from predict.GuidedPredictor import GuidedPredictor
from preprocess.KmerLabelDecoder import KmerLabelDecoder
import numpy as np

class TestGuidedPredictor(TestCase):
    def setUp(self):
        self.model = MockStatefulGapPredictModel()
        self.predictor = GuidedPredictor(self.model)
        self.sequence = "AGTCCGACG"
        self.prediction_length = 2
        self.min_seed_length = 7
        self.decoder = KmerLabelDecoder()

    def test_regenerate_sequence(self):
        sequence, probabilities = self.predictor.regenerate_sequence(self.min_seed_length, self.sequence)
        self.assertEqual(len(probabilities), 2)
        self.assertEqual(sequence, "AGTCCGAAC")
        np.testing.assert_allclose(probabilities[0], np.array([np.exp(-0.3), np.exp(-0.4), np.exp(-1), np.exp(-4)]), atol=0.1)
        np.testing.assert_allclose(probabilities[1], np.array([np.exp(-0.8), np.exp(-0.7), np.exp(-0.9), np.exp(-1)]), atol=0.1)