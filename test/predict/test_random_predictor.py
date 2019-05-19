from unittest import TestCase
from test.predict.MockStatefulGapPredictModel import MockStatefulGapPredictModel
from predict.RandomPredictor import RandomPredictor
from preprocess.KmerLabelDecoder import KmerLabelDecoder
import numpy as np


class TestRandomPredictor(TestCase):
    def setUp(self):
        self.model = MockStatefulGapPredictModel()
        self.predictor = RandomPredictor(self.model)
        self.prediction_length = 2
        self.seed = "AGTCCGA"
        self.length_to_predict = 2
        self.decoder = KmerLabelDecoder()

    def test_predict_random_sequence(self):
        np.random.seed(1)
        sequence, probabilities = self.predictor.predict_random_sequence(self.seed, self.length_to_predict)
        np.random.seed(1)
        sequence1, probabilities1 = self.predictor.predict_random_sequence(self.seed, self.length_to_predict)

        np.testing.assert_allclose(probabilities, probabilities1, atol=0.1)
        self.assertEqual(sequence, sequence1)

        np.random.seed(0)
        sequence2, probabilities2 = self.predictor.predict_random_sequence(self.seed, self.length_to_predict)
        np.testing.assert_raises(AssertionError, np.testing.assert_allclose, probabilities, probabilities2, atol=0.1)
        np.testing.assert_raises(AssertionError, self.assertEqual, sequence, sequence2)

