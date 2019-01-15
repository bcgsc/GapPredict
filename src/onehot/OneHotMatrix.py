import numpy as np

from constants import RnnEncodingConstants as RNN_CONSTANTS
from exceptions.NegativePredictionLengthException import NegativePredictionLengthException
from exceptions.NonpositiveLengthException import NonpositiveLengthException


class _OneHotMatrixUtil:
    def __init__(self, sequence_length, encoding_constants=RNN_CONSTANTS):
        if sequence_length < 1:
            raise NonpositiveLengthException
        self.sequence_length = sequence_length
        self.constants = encoding_constants


class OneHotMatrixDecoder(_OneHotMatrixUtil):
    def __init__(self, sequence_length, encoding_constants=RNN_CONSTANTS):
        super().__init__(sequence_length, encoding_constants)

    def decode_sequences(self, encoded_sequences):
        sequences = self.constants.REVERSE_INTEGER_ENCODING[np.argmax(encoded_sequences, axis=2)]
        return sequences


class OneHotMatrixEncoder(_OneHotMatrixUtil):
    def __init__(self, sequence_length, bases_to_predict=0, encoding_constants=RNN_CONSTANTS):
        #TODO: consider removing bases_to_predict since we aren't using placeholders anymore
        if bases_to_predict < 0:
            raise NegativePredictionLengthException
        super().__init__(sequence_length, encoding_constants)
        self.bases_to_predict = bases_to_predict

    def encode_sequences(self, integer_encoding):
        if len(integer_encoding) == 0:
            return np.array([])

        ONE_HOT_ENCODING = self.constants.ONE_HOT_ENCODING
        encoding_length = ONE_HOT_ENCODING.shape[1]

        cube = ONE_HOT_ENCODING[integer_encoding]

        if self.bases_to_predict > 0:
            extra_cube_shape = (len(integer_encoding), self.bases_to_predict, encoding_length)
            extra_cube = np.zeros(extra_cube_shape, dtype="int8")
            cube = np.concatenate((cube, extra_cube), axis=1)

        return cube
