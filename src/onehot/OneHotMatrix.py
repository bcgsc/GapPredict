import numpy as np
from sklearn import preprocessing

from exceptions.NegativePredictionLengthException import NegativePredictionLengthException
from exceptions.NonpositiveLengthException import NonpositiveLengthException

BASE_ENCODING_IDX_MAP = {
    "A": 1,
    "C": 2,
    "G": 3,
    "T": 4,
    "!": 0
}

class _OneHotMatrixUtil:
    def __init__(self, sequence_length):
        if sequence_length < 1:
            raise NonpositiveLengthException
        self.sequence_length = sequence_length
        self.labeler = preprocessing.LabelEncoder()
        self.encoder = preprocessing.OneHotEncoder(sparse=False, categories="auto")
        self._train_labeler()
        self._train_encoder()

    def _train_labeler(self):
        encoder_train_vector = ["!", "A", "C", "G", "T"]
        self.labeler.fit(encoder_train_vector)

    def _train_encoder(self):
        encoder_train_matrix = np.array([
            [0],
            [1],
            [2],
            [3],
            [4]
        ])
        self.encoder.fit(encoder_train_matrix)


class OneHotMatrixDecoder(_OneHotMatrixUtil):
    def __init__(self, sequence_length):
        super().__init__(sequence_length)

    def decode_sequences(self, encoded_sequences):
        sequences = []
        for sequence_matrix in encoded_sequences:
            decoded_int_vector = self.encoder.inverse_transform(sequence_matrix).reshape(self.sequence_length)
            decoded_bases = self.labeler.inverse_transform(decoded_int_vector)
            sequences.append(decoded_bases)
        return sequences


class OneHotMatrixEncoder(_OneHotMatrixUtil):
    def __init__(self, sequence_length, bases_to_predict=0):
        if bases_to_predict < 0:
            raise NegativePredictionLengthException
        super().__init__(sequence_length)
        self.bases_to_predict = bases_to_predict

    def _encoding_idx(self, base):
        return BASE_ENCODING_IDX_MAP[base]

    def _encode_base(self, char, quality):
        encoding_vector = np.zeros(self.encoding_length)
        if quality is not None:
            encoding_vector[5] = quality
        encoding_vector[self._encoding_idx(char)] = 1
        return encoding_vector

    def encode_sequences(self, sequences, qualities=None):
        has_qualities = True if qualities is not None else False
        self.encoding_length = len(BASE_ENCODING_IDX_MAP) + 1 if has_qualities else len(BASE_ENCODING_IDX_MAP)
        total_sequence_length = self.sequence_length + self.bases_to_predict
        num_sequences = len(sequences)
        cube = np.zeros((num_sequences, total_sequence_length, self.encoding_length))
        for i in range(num_sequences):
            sequence_vector = sequences[i]
            for j in range(len(sequence_vector)):
                char = sequence_vector[j]
                quality = qualities[i][j] if has_qualities else None
                encoded_vector = self._encode_base(char, quality)
                cube[i][j] = encoded_vector
        return cube
