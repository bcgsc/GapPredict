import keras as keras
import numpy as np

import constants.EncodingConstants as CONSTANTS
from onehot.OneHotVector import OneHotVectorDecoder
from predict.new_rnn.SingleLSTMModel import SingleLSTMModel
from preprocess.KmerLabelEncoder import KmerLabelEncoder


class ValidationMetric(keras.callbacks.Callback):
    def __init__(self, reference_seq, min_seed_length, spacing, embedding_dim, latent_dim, epochs, early_stopping=False):
        self.validation_model = SingleLSTMModel(min_seed_length=min_seed_length, stateful=True, batch_size=1, embedding_dim=embedding_dim, latent_dim=latent_dim,
                                with_gpu=True)
        self.reference_seq = reference_seq
        self.min_seed_length = min_seed_length
        self.spacing = spacing
        self.num_epochs = epochs
        self.data = np.zeros(self.num_epochs)
        self.epochs = np.arange(self.num_epochs)
        self.early_stopping = early_stopping
        if self.early_stopping:
            self.model_weight_checkpoint = None
            self.patience = 300
            self.best_epoch = -1

    def _transfer_model_weights(self):
        self.validation_model.set_weights(self.model.get_weights())

    def on_epoch_end(self, epoch, logs=None):
        self.validation_model.reset_states()
        self._transfer_model_weights()
        validation_metric = self._percentage_until_mismatch()
        self.data[epoch] = validation_metric
        if self.early_stopping:
            if self.best_epoch < 0:
                best_model_metric_so_far = -1
            else:
                best_model_metric_so_far = self.data[self.best_epoch]
            #TODO: this will pick the first model with the highest validation metric but it seems pretty easy to reach 1.0 evenutally
            #TODO: perhaps it is worth also storing the weights for the model that reaches 1.0 and hangs onto it for the longest epochs to
            #TODO: compare, it tests whether a stable model is better than a model least likely to overfit
            if validation_metric > best_model_metric_so_far:
                self.model_weight_checkpoint = self.model.get_weights()
                self.best_epoch = epoch
            elif epoch > self.best_epoch + self.patience:
                self.model.stop_training=True

    def on_train_end(self, logs=None):
        self.model.set_weights(self.model_weight_checkpoint)

    def get_data(self):
        return self.data, self.epochs

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
        length = self.min_seed_length

        current_sequence = str(start_string)
        seed = current_sequence[0:length - 1]
        input_seq = label_encoder.encode_kmers([seed], [], with_shifted_output=False)[0]
        self.validation_model.predict(input_seq)
        while remaining_length > 0:
            base = current_sequence[length-1:length]
            base_encoding = label_encoder.encode_kmers([base], [], with_shifted_output=False)[0]

            prediction = self.validation_model.predict(base_encoding)
            decoded_prediction = one_hot_decoder.decode_sequences(prediction)[0][0]
            bases_predicted = bases_to_predict - remaining_length
            expected_base = string_to_predict[bases_predicted]
            if decoded_prediction != expected_base:
                break;

            current_sequence += decoded_prediction
            remaining_length -= 1
            length += 1
        return bases_predicted/bases_to_predict



