import numpy as np

from constants import RnnEncodingConstants as CONSTANTS


class SequenceSplitEncoder:
    def __init__(self, split_idx):
        self.split_idx = max(split_idx, 1)

    def split_sequences(self, reads):
        ONE_HOT_ENCODING_IDX_MAP = CONSTANTS.INTEGER_ENCODING_MAP

        min_length = min(list(map(lambda x:len(x), reads)))
        min_split_idx = min(min_length - 1, self.split_idx)
        num_reads = len(reads)
        input_characters = min_split_idx
        output_characters = min_length - input_characters

        input = np.zeros((num_reads, input_characters))
        output = np.zeros((num_reads, output_characters))
        shifted_output = np.zeros((num_reads, output_characters))

        for i in range(num_reads):
            read = reads[i]
            for j in range(input_characters):
                encoding = ONE_HOT_ENCODING_IDX_MAP[read[j]]
                input[i][j] = encoding

            for j in range(output_characters):
                encoding = ONE_HOT_ENCODING_IDX_MAP[read[min_split_idx + j]]
                output[i][j] = encoding

            shifted_output[i][0] = ONE_HOT_ENCODING_IDX_MAP["!"]
        shifted_output[:, 1:output_characters] = output[:, 0:output_characters - 1]

        return input, output, shifted_output