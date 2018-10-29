import numpy as np

from constants import EncodingConstants as CONSTANTS


class KmerLabelEncoder:
    def __init__(self):
        pass

    # TODO: consider removing this parameter once we don't need it (eg. we decide on if we want to pad or not)
    def encode_kmers(self, input_kmers, output_kmers, quality_kmers, fill_in_the_blanks=False):
        #TODO: handle empty arrays
        ONE_HOT_ENCODING_IDX_MAP = CONSTANTS.INTEGER_ENCODING_MAP

        input_kmer_count = len(input_kmers)
        input_kmer_length = len(input_kmers[0])
        input_shape = (input_kmer_count, input_kmer_length)
        input_seq = np.zeros(input_shape, dtype="int8")

        input_quality = np.array(quality_kmers)

        output_kmer_count = len(output_kmers)
        output_kmer_length = len(output_kmers[0])
        full_output_kmer_length = (input_kmer_length + output_kmer_length) if fill_in_the_blanks else output_kmer_length
        output_shape = (output_kmer_count, full_output_kmer_length)
        output_seq = np.zeros(output_shape, dtype="int8")
        shifted_output_seq = np.zeros(output_shape, dtype="int8")

        for i in range(input_kmer_count):
            kmer = input_kmers[i]
            for j in range(input_kmer_length):
                encoding = ONE_HOT_ENCODING_IDX_MAP[kmer[j]]
                input_seq[i][j] = encoding
                if fill_in_the_blanks:
                    output_seq[i][j] = encoding
                    shifted_output_seq[i][j+1] = encoding #TODO: assumes theres at least 1 output base

        for i in range(output_kmer_count):
            kmer = output_kmers[i]
            for j in range(output_kmer_length):
                encoding = ONE_HOT_ENCODING_IDX_MAP[kmer[j]]
                if fill_in_the_blanks:
                    column = input_kmer_length + j
                    output_seq[i][column] = encoding
                    if column + 1 < full_output_kmer_length:
                        shifted_output_seq[i][column + 1] = encoding
                else:
                    output_seq[i][j] = encoding
                    if j + 1 < full_output_kmer_length:
                        shifted_output_seq[i][j + 1] = encoding

        return input_seq, np.array(input_quality), output_seq, shifted_output_seq
