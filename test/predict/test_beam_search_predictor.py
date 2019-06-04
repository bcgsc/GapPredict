from unittest import TestCase
from test.predict.MockStatelessGapPredictModel import MockStatelessGapPredictModel
from predict.BeamSearchPredictor import BeamSearchPredictor
from preprocess.KmerLabelDecoder import KmerLabelDecoder
import numpy as np

class TestBeamSearchPredictor(TestCase):
    def setUp(self):
        self.model = MockStatelessGapPredictModel()
        self.predictor = BeamSearchPredictor(self.model)
        self.seed = "AGTCCGA"
        self.prediction_length = 2
        self.decoder = KmerLabelDecoder()

    def test_predict_next_n_bases(self):
        beam_length = 2
        sequences, probabilities = self.predictor.predict_next_n_bases(self.seed, self.prediction_length, beam_length)
        np.testing.assert_allclose(probabilities, np.array([-1.1, -1.2]), atol=0.1)

        self.assertEqual(len(sequences), 2)
        decoded_prediction1 = self.decoder.decode(sequences[0])
        decoded_prediction2 = self.decoder.decode(sequences[1])
        self.assertEqual("AGTCCGACA", decoded_prediction1)
        self.assertEqual("AGTCCGACC", decoded_prediction2)

    def test_predict_next_n_bases_greedy(self):
        sequence, probability = self.predictor.predict_next_n_bases_greedy(self.seed, self.prediction_length)
        self.assertAlmostEqual(probability, -1.6, 1)

        decoded_prediction = self.decoder.decode(sequence)
        self.assertEqual("AGTCCGAAA", decoded_prediction)
