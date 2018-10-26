import numpy as np

from exceptions.NegativePredictionLengthException import NegativePredictionLengthException
from exceptions.NonpositiveLengthException import NonpositiveLengthException

BASE_ENCODING_IDX_MAP = {
    "A": 1,
    "C": 2,
    "G": 3,
    "T": 4,
    "!": 0
}
REVERSE_ENCODING = np.array(["!", "A", "C", "G", "T"])

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
        for sequence_matrix in encoded_sequences:
            decoded_int_vector = np.argmax(sequence_matrix, axis=1)
            decoded_bases = REVERSE_ENCODING[decoded_int_vector[:]]
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

    def encode_sequences(self, sequences, qualities=None):
        has_qualities = True if qualities is not None else False
        encoding_length = len(BASE_ENCODING_IDX_MAP) + 1 if has_qualities else len(BASE_ENCODING_IDX_MAP)
        total_sequence_length = self.sequence_length + self.bases_to_predict
        num_sequences = len(sequences)

        cube = np.zeros((num_sequences, total_sequence_length, encoding_length))
        for i in range(num_sequences):
            sequence_vector = sequences[i]
            cube_matrix = cube[i]
            quality_vector = qualities[i] if has_qualities else None
            for j in range(len(sequence_vector)):
                char = sequence_vector[j]
                cube_vector = cube_matrix[j]
                cube_vector[self._encoding_idx(char)] = 1
                if has_qualities:
                    cube_vector[encoding_length - 1] = quality_vector[j]
        return cube
