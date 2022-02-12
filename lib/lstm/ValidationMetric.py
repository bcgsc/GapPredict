import math

import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import categorical_crossentropy

import constants.EncodingConstants as CONSTANTS
from onehot.OneHotVector import OneHotVectorDecoder
from onehot.OneHotVector import OneHotVectorEncoder
from lstm.GapPredictModel import GapPredictModel
from preprocess.KmerLabelEncoder import KmerLabelEncoder
from preprocess.SequenceMatchCalculator import SequenceMatchCalculator


class ValidationMetric(Callback):
    def __init__(self, reference_seqs, min_seed_length, spacing, embedding_dim, latent_dim, epochs, patience=200, early_stopping=False):
        self.reference_seqs = self._init_reference_characters(reference_seqs)
        self.lengths = np.array(list(map(lambda x:len(x), self.reference_seqs)))
        self.min_seed_length = min_seed_length
        self.spacing = spacing
        self.num_epochs = epochs
        self.accuracy_data = []
        self.loss_data = []
        self.early_stopping = early_stopping
        self.one_hot_vector_encoder = OneHotVectorEncoder(1, encoding_constants=CONSTANTS)
        self.label_encoder = KmerLabelEncoder()
        self.validator = SequenceMatchCalculator()

        self.equal_lengths = self._equal_length_sequences()
        if self.equal_lengths:
            self.encoded_ref_seqs = self.label_encoder.encode_kmers(self.reference_seqs, [], with_shifted_output=False)[0]
        if self.early_stopping:
            self.model_weight_checkpoint = None
            self.patience = patience
            self.best_epoch = -1

        batch_size = len(self.reference_seqs) if self.equal_lengths else 1
        self.validation_model = GapPredictModel(min_seed_length=min_seed_length, stateful=True, batch_size=batch_size, embedding_dim=embedding_dim, latent_dim=latent_dim,
                                                with_gpu=True)

    def _init_reference_characters(self, reference_seqs):
        reference_characters = []
        for i in range(len(reference_seqs)):
            reference_characters.append(np.array(list(reference_seqs[i])))
        return reference_characters

    def _equal_length_sequences(self):
        lengths = list(map(lambda x: len(x), self.reference_seqs))
        lengths_set = set()
        for length in lengths:
            lengths_set.add(length)
        return len(lengths_set) <= 1

    def _transfer_model_weights(self):
        self.validation_model.set_weights(self.model.get_weights())

    def _weighted_mean(self, metrics):
        return np.sum(metrics*self.lengths)/np.sum(self.lengths)

    def _categorical_cross_entropy_fast(self, basewise_probabilities):
        total_seqs = len(self.reference_seqs)
        offset = self.min_seed_length + self.spacing
        seq_length = len(self.reference_seqs[0])

        y = np.zeros((total_seqs, seq_length - offset, len(CONSTANTS.ONE_HOT_ENCODING)))

        for i in range(total_seqs):
            y[i] = self.one_hot_vector_encoder.encode_sequences(self.encoded_ref_seqs[i][offset:])

        yhat = basewise_probabilities
        loss = K.eval(categorical_crossentropy(K.variable(y), K.variable(yhat)))
        all_loss = np.mean(loss, axis=1)
        return all_loss

    def _categorical_cross_entropy_slow(self, basewise_probabilities):
        total_seqs = len(self.reference_seqs)
        offset = self.min_seed_length + self.spacing
        all_loss = np.zeros(total_seqs)

        for i in range(total_seqs):
            seq = self.reference_seqs[i]
            characters = seq[offset:]
            y = self.one_hot_vector_encoder.encode_sequences(self.label_encoder.encode_kmers(np.array([]), characters, with_shifted_output=False)[1])
            yhat = basewise_probabilities[i]
            loss = K.eval(categorical_crossentropy(K.variable(y), K.variable(yhat)))
            all_loss[i] = np.mean(loss)
        return all_loss

    def _categorical_cross_entropy(self, basewise_probabilities):
        if self.equal_lengths:
            return self._categorical_cross_entropy_fast(basewise_probabilities)
        else:
            return self._categorical_cross_entropy_slow(basewise_probabilities)

    def on_epoch_end(self, epoch, logs=None):
        self.validation_model.reset_states()
        self._transfer_model_weights()
        validation_metrics, basewise_probabilities = self._percentage_matched()
        loss = self._categorical_cross_entropy(basewise_probabilities)
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

    def _percentage_matched_slow(self):
        prediction_length = 1
        one_hot_decoder = OneHotVectorDecoder(prediction_length, encoding_constants=CONSTANTS)
        total_seqs = len(self.reference_seqs)
        percentages = np.zeros(total_seqs)

        basewise_probabilities = []

        known_length = self.min_seed_length + self.spacing
        for i in range(total_seqs):
            reference_seq = self.reference_seqs[i]
            sequence_length = len(reference_seq)
            encoded_reference_seq = self.label_encoder.encode_kmers([reference_seq], [], with_shifted_output=False)[0]

            remaining_length = sequence_length - known_length
            length = self.min_seed_length

            model_predictions = np.zeros((remaining_length, len(CONSTANTS.ONE_HOT_ENCODING)))

            current_sequence = np.empty(sequence_length, dtype="<U1")
            current_sequence[:known_length] = reference_seq[:known_length]

            seed = encoded_reference_seq[:, 0:length - 1]

            self.validation_model.reset_states()
            self.validation_model.predict(seed)
            while remaining_length > 0:
                next_base_encoding = encoded_reference_seq[:, length - 1:length]

                prediction = self.validation_model.predict(next_base_encoding)
                decoded_prediction = one_hot_decoder.decode_sequences(prediction)[0][0]

                model_predictions[length - known_length] = prediction[0]

                current_sequence[length] = decoded_prediction
                remaining_length -= 1
                length += 1
            matches = self.validator.compare_sequences(current_sequence, reference_seq)[known_length:]
            percentages[i] = np.mean(matches)

            basewise_probabilities.append(model_predictions)
        return percentages, basewise_probabilities

    def _percentage_matched_fast(self):
        prediction_length = 1
        one_hot_decoder = OneHotVectorDecoder(prediction_length, encoding_constants=CONSTANTS)
        total_seqs = len(self.encoded_ref_seqs)

        known_length = self.min_seed_length + self.spacing
        sequence_length = len(self.encoded_ref_seqs[0])
        remaining_length = sequence_length - known_length
        length = self.min_seed_length

        basewise_probabilities = np.zeros((total_seqs, remaining_length, len(CONSTANTS.ONE_HOT_ENCODING)))
        predicted_sequences = np.empty((total_seqs, sequence_length), dtype="<U1")
        for i in range(len(self.reference_seqs)):
            predicted_sequences[i][:known_length] = self.reference_seqs[i][:known_length]

        seeds = self.encoded_ref_seqs[:,0:length - 1]
        self.validation_model.reset_states()
        self.validation_model.predict(seeds)

        while remaining_length > 0:
            next_base_encodings = self.encoded_ref_seqs[:, length - 1 : length]
            prediction = self.validation_model.predict(next_base_encodings)
            decoded_predictions = one_hot_decoder.decode_sequences(prediction)

            basewise_probabilities[:, length - known_length] = prediction

            predicted_sequences[:,length] = decoded_predictions[:, 0]

            remaining_length -= 1
            length += 1
        matches = self.validator.compare_sequences(predicted_sequences, self.reference_seqs)[:, known_length:]
        percentages = np.mean(matches, axis=1)
        return percentages, basewise_probabilities

    def _percentage_matched(self):
        if self.equal_lengths:
            return self._percentage_matched_fast()
        else:
            return self._percentage_matched_slow()


