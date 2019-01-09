import numpy as np

from constants import EncodingConstants as CONSTANTS
from constants import RnnEncodingConstants as RNN_CONSTANTS
from exceptions.NegativePredictionLengthException import NegativePredictionLengthException
from exceptions.NonpositiveLengthException import NonpositiveLengthException


class _OneHotMatrixUtil:
    def __init__(self, sequence_length, use_rnn_constants=True):
        if sequence_length < 1:
            raise NonpositiveLengthException
        self.sequence_length = sequence_length
        self.constants = RNN_CONSTANTS if use_rnn_constants else CONSTANTS


class OneHotMatrixDecoder(_OneHotMatrixUtil):
    def __init__(self, sequence_length, use_rnn_constants=True):
        super().__init__(sequence_length, use_rnn_constants)

    def decode_sequences(self, encoded_sequences):
        sequences = self.constants.REVERSE_INTEGER_ENCODING[np.argmax(encoded_sequences, axis=2)]
        return sequences


class OneHotMatrixEncoder(_OneHotMatrixUtil):
    def __init__(self, sequence_length, bases_to_predict=0, use_rnn_constants=True):
        #TODO: consider removing bases_to_predict since we aren't using placeholders anymore
        if bases_to_predict < 0:
            raise NegativePredictionLengthException
        super().__init__(sequence_length, use_rnn_constants)
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
