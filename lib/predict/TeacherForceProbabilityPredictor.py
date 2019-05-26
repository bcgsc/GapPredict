from predict.GuidedPredictor import GuidedPredictor
from preprocess.KmerLabelEncoder import KmerLabelEncoder
import numpy as np

class TeacherForceProbabilityPredictor:
    def __init__(self, stateful_model):
        self.predictor = GuidedPredictor(stateful_model)

    def get_probabilities(self, min_seed_length, full_sequence):
        basewise_probabilities = self.predictor.regenerate_sequence(min_seed_length, full_sequence)[1]
        predicted_sequence = full_sequence[min_seed_length:]
        encoder = KmerLabelEncoder()
        encoded_predicted_sequence = encoder.encode_kmers([predicted_sequence], [], False)[0][0]

        predicted_sequence_length = len(encoded_predicted_sequence)
        probabilities = np.zeros(predicted_sequence_length)

        for i in range(predicted_sequence_length):
            probabilities[i] = basewise_probabilities[i][encoded_predicted_sequence[i]]

        return probabilities



