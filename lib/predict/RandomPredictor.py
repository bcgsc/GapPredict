import numpy as np

import constants.EncodingConstants as CONSTANTS
from preprocess.KmerLabelEncoder import KmerLabelEncoder

class RandomPredictor:
    def __init__(self, stateful_model):
        self.model = stateful_model

    def predict_random_sequence(self, seed, length_to_predict):
        label_encoder = KmerLabelEncoder()

        probability_vector = np.zeros(length_to_predict)
        remaining_length = length_to_predict

        self.model.reset_states()
        current_sequence = str(seed)

        input_seq = label_encoder.encode_kmers([current_sequence[:len(current_sequence) - 1]], [], with_shifted_output=False)[0]
        self.model.predict(input_seq)
        while remaining_length > 0:
            base = current_sequence[-1]
            base_encoding = label_encoder.encode_kmers([base], [], with_shifted_output=False)[0]

            prediction = self.model.predict(base_encoding)
            random_base_idx = np.random.randint(len(CONSTANTS.ONE_HOT_ENCODING))
            random_base = CONSTANTS.REVERSE_INTEGER_ENCODING[random_base_idx]
            current_sequence += random_base
            probability_vector[length_to_predict - remaining_length] = prediction[0][random_base_idx]

            remaining_length -= 1
        return current_sequence, probability_vector