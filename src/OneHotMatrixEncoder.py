import numpy as np
from sklearn import preprocessing

from exceptions.NonpositiveLengthException import NonpositiveLengthException


class OneHotMatrixEncoder:
    def __init__(self, input_length):
        if input_length < 1:
            raise NonpositiveLengthException
        self.input_length = input_length
        self.labeler = preprocessing.LabelEncoder()
        self.encoder = preprocessing.OneHotEncoder(sparse=False, categories="auto")
        self._train_labeler()
        self._train_encoder()

    def _train_labeler(self):
        encoder_train_vector = ["A", "C", "G", "T"]
        self.labeler.fit(encoder_train_vector)

    def _train_encoder(self):
        encoder_train_matrix = np.array([
            [0],
            [1],
            [2],
            [3]
        ])
        self.encoder.fit(encoder_train_matrix)

    def encode_sequences(self, sequences, qualities=None):
        has_qualities = True if qualities is not None else False
        encoding_length = 5 if has_qualities else 4
        num_sequences = len(sequences)
        cube = np.zeros((num_sequences, self.input_length, encoding_length))
        for i in range(num_sequences):
            sequence_vector = sequences[i]
            int_column_vector = self.labeler.transform(sequence_vector).reshape(self.input_length, 1)
            one_hot_matrix = self.encoder.transform(int_column_vector)

            if has_qualities:
                quality_vector = qualities[i]
                quality_column_vector = quality_vector.reshape(self.input_length, 1)
                one_hot_matrix = np.append(one_hot_matrix, quality_column_vector, axis=1)
                #TODO: there are several ways to append columns, this is probably a slow way so rethink this later

            cube[i] = one_hot_matrix
        return cube

    def decode_sequences(self, encoded_sequences):
        sequences = []
        for sequence_matrix in encoded_sequences:
            decoded_int_vector = self.encoder.inverse_transform(sequence_matrix).reshape(self.input_length)
            decoded_bases = self.labeler.inverse_transform(decoded_int_vector)
            sequences.append(decoded_bases)
        return sequences