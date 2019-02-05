import numpy as np
import keras as keras
import constants.EncodingConstants as CONSTANTS
from onehot.OneHotVector import OneHotVectorDecoder
from preprocess.KmerLabelEncoder import KmerLabelEncoder

class ValidationMetric(keras.callbacks.Callback):
    #TODO spacing
    def __init__(self, model, reference_seq, min_seed_length, spacing):
        self.model = model
        self.reference_seq = reference_seq
        self.min_seed_length = min_seed_length
        self.spacing = spacing
        self.data = []
        self.epochs = []

    def on_epoch_end(self, epoch, logs=None):
        validation_metric = self._percentage_until_mismatch()
        self.data.append(validation_metric)
        self.epochs.append(epoch)

    def get_data(self):
        return np.array(self.data), np.array(self.epochs)

    def _percentage_until_mismatch(self):
        label_encoder = KmerLabelEncoder()
        prediction_length = 1
        one_hot_decoder = OneHotVectorDecoder(prediction_length, encoding_constants=CONSTANTS)

        sequence_length = len(self.reference_seq)
        known_length = self.min_seed_length + self.spacing
        start_string = self.reference_seq[0:known_length]
        string_to_predict = self.reference_seq[known_length:]

        bases_to_predict = sequence_length - known_length

        remaining_length = bases_to_predict
        seed_length = self.min_seed_length

        current_sequence = str(start_string)
        while remaining_length > 0:
            seed = current_sequence[0:seed_length]
            input_seq = label_encoder.encode_kmers([seed], [], with_shifted_output=False)[0]

            prediction = self.model.predict(input_seq)
            decoded_prediction = one_hot_decoder.decode_sequences(prediction)[0][0]
            bases_predicted = bases_to_predict - remaining_length
            expected_base = string_to_predict[bases_predicted]
            if decoded_prediction != expected_base:
                break;

            current_sequence += decoded_prediction
            remaining_length -= 1
            seed_length += 1
        return bases_predicted/bases_to_predict



