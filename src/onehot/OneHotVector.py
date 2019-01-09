import numpy as np

from constants import EncodingConstants as CONSTANTS
from exceptions.NonpositiveLengthException import NonpositiveLengthException
from onehot.OneHotMatrix import OneHotMatrixDecoder, OneHotMatrixEncoder


class _OneHotVectorUtil:
    def __init__(self, sequence_length):
        if sequence_length < 1:
            raise NonpositiveLengthException
        self.sequence_length = sequence_length


class OneHotVectorDecoder(_OneHotVectorUtil):
    def __init__(self, sequence_length):
        super().__init__(sequence_length)
        self.decoder = OneHotMatrixDecoder(sequence_length, use_rnn_constants=False)

    def decode_sequences(self, encoded_sequences):
        dimensions = (len(encoded_sequences), self.sequence_length, len(CONSTANTS.ONE_HOT_ENCODING))
        sequences = self.decoder.decode_sequences(encoded_sequences.reshape(dimensions))
        return sequences


class OneHotVectorEncoder(_OneHotVectorUtil):
    def __init__(self, sequence_length):
        super().__init__(sequence_length)
        self.encoder = OneHotMatrixEncoder(sequence_length, use_rnn_constants=False)

    def encode_sequences(self, integer_encoding):
        if len(integer_encoding) == 0:
            return np.array([])

        cube = self.encoder.encode_sequences(integer_encoding)
        encoding_length = len(CONSTANTS.ONE_HOT_ENCODING)
        dimensions = (len(cube), encoding_length * self.sequence_length)
        encoded_matrix = cube.reshape(dimensions)

        return encoded_matrix
