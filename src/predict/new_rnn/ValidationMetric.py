import math

import numpy as np
from keras import backend as K
from keras.callbacks import Callback
from keras.losses import categorical_crossentropy

import constants.EncodingConstants as CONSTANTS
from onehot.OneHotVector import OneHotVectorDecoder
from onehot.OneHotVector import OneHotVectorEncoder
from predict.new_rnn.SingleLSTMModel import SingleLSTMModel
from preprocess.KmerLabelEncoder import KmerLabelEncoder
from preprocess.SequenceMatchCalculator import SequenceMatchCalculator


class ValidationMetric(Callback):
    def __init__(self, reference_seqs, min_seed_length, spacing, embedding_dim, latent_dim, epochs, patience=200, early_stopping=False):
        self.validation_model = SingleLSTMModel(min_seed_length=min_seed_length, stateful=True, batch_size=1, embedding_dim=embedding_dim, latent_dim=latent_dim,
                                with_gpu=True)
        self.reference_seqs = reference_seqs
        self.lengths = np.array(list(map(lambda x:len(x), self.reference_seqs)))
        self.min_seed_length = min_seed_length
        self.spacing = spacing
        self.num_epochs = epochs
        self.accuracy_data = []
        self.loss_data = []
        self.early_stopping = early_stopping
        self.one_hot_encoder = OneHotVectorEncoder(1, encoding_constants=CONSTANTS)
        self.label_encoder = KmerLabelEncoder()
        self.validator = SequenceMatchCalculator()
        if self.early_stopping:
            self.model_weight_checkpoint = None
            self.patience = patience
            self.best_epoch = -1

    def _transfer_model_weights(self):
        self.validation_model.set_weights(self.model.get_weights())

    def _weighted_mean(self, metrics):
        return np.sum(metrics*self.lengths)/np.sum(self.lengths)

    def categorical_cross_entropy(self, basewise_probabilities):
        total_seqs = len(self.reference_seqs)
        offset = self.min_seed_length + self.spacing
        all_loss = np.zeros(total_seqs)

        for i in range(total_seqs):
            seq = self.reference_seqs[i]
            characters = np.array(list(seq[offset:]))
            y = self.one_hot_encoder.encode_sequences(self.label_encoder.encode_kmers(np.array([]), characters, with_shifted_output=False)[1])
            yhat = basewise_probabilities[i]
            loss = K.eval(categorical_crossentropy(K.variable(y), K.variable(yhat)))
            all_loss[i] = np.mean(loss)
        return all_loss #TODO: verify that mean loss is correct, also verify that model.evaluate does the exact same thing


    def on_epoch_end(self, epoch, logs=None):
        self.validation_model.reset_states()
        self._transfer_model_weights()
        validation_metrics, basewise_probabilities = self._percentage_matched()
        loss = self.categorical_cross_entropy(basewise_probabilities) #TODO: need to save this somewhere, and use it to do early stopping
        self.accuracy_data.append(validation_metrics)
        self.loss_data.append(loss)
        if self.early_stopping:
            if self.best_epoch < 0:
                best_mean_model_metric_so_far = math.inf
            else:
                best_mean_model_metric_so_far = self._weighted_mean(self.loss_data[self.best_epoch])
            mean_current_metric = self._weighted_mean(loss)
            if mean_current_metric < best_mean_model_metric_so_far:
                self.model_weight_checkpoint = self.model.get_weights()
                self.best_epoch = epoch
            elif epoch > self.best_epoch + self.patience:
                self.model.stop_training = True

    def on_train_end(self, logs=None):
        self.model.set_weights(self.model_weight_checkpoint)

    def get_data(self):
        return np.array(self.accuracy_data), np.array(self.loss_data)

    def get_best_epoch(self):
        return self.best_epoch

    def _percentage_matched(self):
        prediction_length = 1
        one_hot_decoder = OneHotVectorDecoder(prediction_length, encoding_constants=CONSTANTS)
        total_seqs = len(self.reference_seqs)
        percentages = np.zeros(total_seqs)

        basewise_probabilities = []

        for i in range(total_seqs):
            reference_seq = self.reference_seqs[i]
            sequence_length = len(reference_seq)
            known_length = self.min_seed_length + self.spacing
            start_string = reference_seq[0:known_length]

            remaining_length = sequence_length - known_length
            length = self.min_seed_length

            model_predictions = np.zeros((remaining_length, len(CONSTANTS.ONE_HOT_ENCODING)))

            current_sequence = str(start_string)
            seed = reference_seq[0:length - 1]
            input_seq = self.label_encoder.encode_kmers([seed], [], with_shifted_output=False)[0]

            self.validation_model.reset_states()
            self.validation_model.predict(input_seq)
            while remaining_length > 0:
                base = reference_seq[length-1:length]
                base_encoding = self.label_encoder.encode_kmers([base], [], with_shifted_output=False)[0]

                prediction = self.validation_model.predict(base_encoding)
                decoded_prediction = one_hot_decoder.decode_sequences(prediction)[0][0]

                model_predictions[length - known_length] = prediction[0]

                current_sequence += decoded_prediction
                remaining_length -= 1
                length += 1
            actual_sequence = reference_seq[known_length:]
            predicted_sequence = current_sequence[known_length:]
            matches = self.validator.compare_sequences(predicted_sequence, actual_sequence)
            percentage_predicted = np.mean(matches)
            percentages[i] = percentage_predicted

            basewise_probabilities.append(model_predictions)
        return percentages, basewise_probabilities


