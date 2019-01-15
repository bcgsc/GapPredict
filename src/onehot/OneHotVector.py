import numpy as np

from constants import EncodingConstants as CONSTANTS
from exceptions.NonpositiveLengthException import NonpositiveLengthException
from onehot.OneHotMatrix import OneHotMatrixDecoder, OneHotMatrixEncoder


class _OneHotVectorUtil:
    def __init__(self, sequence_length, encoding_constants=CONSTANTS):
        if sequence_length < 1:
            raise NonpositiveLengthException
        self.sequence_length = sequence_length
        self.encoding_constants = encoding_constants


class OneHotVectorDecoder(_OneHotVectorUtil):
    def __init__(self, sequence_length, encoding_constants=CONSTANTS):
        super().__init__(sequence_length, encoding_constants=encoding_constants)
        self.decoder = OneHotMatrixDecoder(sequence_length, encoding_constants=self.encoding_constants)

    def decode_sequences(self, encoded_sequences):
        dimensions = (len(encoded_sequences), self.sequence_length, len(self.encoding_constants.ONE_HOT_ENCODING))
        sequences = self.decoder.decode_sequences(encoded_sequences.reshape(dimensions))
        return sequences


class OneHotVectorEncoder(_OneHotVectorUtil):
    def __init__(self, sequence_length, encoding_constants=CONSTANTS):
        super().__init__(sequence_length, encoding_constants=encoding_constants)
        self.encoder = OneHotMatrixEncoder(sequence_length, encoding_constants=self.encoding_constants)

    def encode_sequences(self, integer_encoding):
        if len(integer_encoding) == 0:
            return np.array([])

        cube = self.encoder.encode_sequences(integer_encoding)
        encoding_length = len(self.encoding_constants.ONE_HOT_ENCODING)
        dimensions = (len(cube), encoding_length * self.sequence_length)
        encoded_matrix = cube.reshape(dimensions)

        return encoded_matrix
