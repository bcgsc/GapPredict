import numpy as np

from constants import EncodingConstants as CONSTANTS


class SequenceSplitEncoder:
    def __init__(self, split_idx):
        self.split_idx = max(split_idx, 1)

    def _validate_sequence(self, seq):
        is_valid = True
        for l in seq:
            if l not in CONSTANTS.BASES:
                is_valid = False
                break

        return is_valid


    def split_sequences(self, reads, output_length=None):
        ONE_HOT_ENCODING_IDX_MAP = CONSTANTS.INTEGER_ENCODING_MAP

        min_length = min(list(map(lambda x:len(x), reads)))
        min_split_idx = min(min_length - 1, self.split_idx)
        input_characters = min_split_idx
        output_characters = min_length - input_characters if \
            output_length is None else min(output_length, min_length - input_characters)

        total_length = input_characters+output_characters
        clean_reads = list(filter(lambda x: self._validate_sequence(x), map(lambda x: x[0:total_length], reads)))
        num_reads = len(clean_reads)

        input = np.zeros((num_reads, input_characters), dtype=np.int8)
        output = np.zeros((num_reads, output_characters), dtype=np.int8)

        for i in range(num_reads):
            read = clean_reads[i]
            for j in range(input_characters):
                encoding = ONE_HOT_ENCODING_IDX_MAP[read[j]]
                input[i][j] = encoding

            for j in range(output_characters):
                encoding = ONE_HOT_ENCODING_IDX_MAP[read[min_split_idx + j]]
                output[i][j] = encoding

        return input, output