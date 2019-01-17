import numpy as np

from constants import RnnEncodingConstants as CONSTANTS


class KmerLabelEncoder:
    def __init__(self):
        pass

    def encode_kmers(self, input_kmers, output_kmers, with_shifted_output=True):
        ONE_HOT_ENCODING_IDX_MAP = CONSTANTS.INTEGER_ENCODING_MAP

        input_kmer_count = len(input_kmers)
        if input_kmer_count > 0:
            input_kmer_length = len(input_kmers[0])
            input_shape = (input_kmer_count, input_kmer_length)
            input_seq = np.zeros(input_shape, dtype="int8")

            for i in range(input_kmer_count):
                kmer = input_kmers[i]
                for j in range(input_kmer_length):
                    encoding = ONE_HOT_ENCODING_IDX_MAP[kmer[j]]
                    input_seq[i][j] = encoding
        else:
            input_seq = np.array([])

        output_kmer_count = len(output_kmers)

        if output_kmer_count > 0:
            output_kmer_length = len(output_kmers[0])
            output_shape = (output_kmer_count, output_kmer_length)
            output_seq = np.zeros(output_shape, dtype="int8")

            for i in range(output_kmer_count):
                kmer = output_kmers[i]
                for j in range(output_kmer_length):
                    encoding = ONE_HOT_ENCODING_IDX_MAP[kmer[j]]
                    output_seq[i][j] = encoding

            if with_shifted_output:
                shifted_output_seq = np.zeros(output_shape, dtype="int8")
                for i in range(output_kmer_count):
                    shifted_output_seq[i][0] = ONE_HOT_ENCODING_IDX_MAP["!"]
                shifted_output_seq[:, 1:output_kmer_length] = output_seq[:, 0:output_kmer_length-1]
            else:
                shifted_output_seq = np.array([])
        else:
            output_seq = np.array([])
            shifted_output_seq = np.array([])

        return input_seq, output_seq, shifted_output_seq
