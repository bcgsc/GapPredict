from unittest import TestCase
from test.predict.MockStatefulGapPredictModel import MockStatefulGapPredictModel
from predict.TeacherForceProbabilityPredictor import TeacherForceProbabilityPredictor
from preprocess.KmerLabelDecoder import KmerLabelDecoder
import numpy as np


class TestTeacherForceProbabilityPredictor(TestCase):
    def setUp(self):
        self.model = MockStatefulGapPredictModel()
        self.predictor = TeacherForceProbabilityPredictor(self.model)
        self.sequence = "AGTCCGACG"
        self.prediction_length = 2
        self.min_seed_length = 7
        self.decoder = KmerLabelDecoder()

    def test_get_probabilities(self):
        probabilities = self.predictor.get_probabilities(self.min_seed_length, self.sequence)
        np.testing.assert_allclose(probabilities, np.array([np.exp(-0.4), np.exp(-0.9)]),
                                   atol=0.1)
