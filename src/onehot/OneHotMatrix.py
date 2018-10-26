import numpy as np

from constants import EncodingConstants as CONSTANTS
from exceptions.NegativePredictionLengthException import NegativePredictionLengthException
from exceptions.NonpositiveLengthException import NonpositiveLengthException


class _OneHotMatrixUtil:
    def __init__(self, sequence_length):
        if sequence_length < 1:
            raise NonpositiveLengthException
        self.sequence_length = sequence_length


class OneHotMatrixDecoder(_OneHotMatrixUtil):
    def __init__(self, sequence_length):
        super().__init__(sequence_length)

    def decode_sequences(self, encoded_sequences):
        sequences = []
        REVERSE_ONE_HOT_ENCODING = CONSTANTS.REVERSE_INTEGER_ENCODING
        for sequence_matrix in encoded_sequences:
            decoded_int_vector = np.argmax(sequence_matrix, axis=1)
            decoded_bases = REVERSE_ONE_HOT_ENCODING[decoded_int_vector[:]]
            sequences.append(decoded_bases)
        return sequences


class OneHotMatrixEncoder(_OneHotMatrixUtil):
    def __init__(self, sequence_length, bases_to_predict=0):
        if bases_to_predict < 0:
            raise NegativePredictionLengthException
        super().__init__(sequence_length)
        self.bases_to_predict = bases_to_predict

    def encode_sequences(self, integer_encoding, qualities=None):
        has_qualities = True if qualities is not None else False

        ONE_HOT_ENCODING = CONSTANTS.ONE_HOT_QUALITY_ENCODING if has_qualities else CONSTANTS.ONE_HOT_ENCODING
        encoding_length = ONE_HOT_ENCODING.shape[1]

        cube = ONE_HOT_ENCODING[integer_encoding]

        if has_qualities:
            for i in range(len(qualities)):
                quality_vector = qualities[i]
                for j in range(len(quality_vector)):
                    cube[i][j][encoding_length - 1] = quality_vector[j]

        if self.bases_to_predict > 0:
            extra_cube_shape = (len(integer_encoding), self.bases_to_predict, encoding_length)
            extra_cube = np.zeros(extra_cube_shape, dtype="int8")
            cube = np.concatenate((cube, extra_cube), axis=1)

        return cube
