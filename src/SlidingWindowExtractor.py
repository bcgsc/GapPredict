import numpy as np


class SlidingWindowParamException(Exception):
    pass


class SlidingWindowExtractor:
    def __init__(self, input_length, spacing, output_length):
        if input_length <= 0 or spacing <= 0 or output_length <= 0:
            raise SlidingWindowParamException("Lengths must be positive")

        self.input_length = input_length
        self.spacing = spacing
        self.output_length = output_length
        self.k = self.input_length + self.spacing + self.output_length

    def _calculate_matrix_rows(self, parsed_fastqs):
        rows = 0
        for fastq in parsed_fastqs:
            rows += len(fastq.sequence) - self.k + 1
        return max(rows, 0)

    def extract_input_output_from_sequence(self, parsed_fastqs):
        # TODO: assumes that the sequence and quality are the same length
        rows = self._calculate_matrix_rows(parsed_fastqs)
        input_seq = []
        input_quality = np.zeros((rows, self.input_length))
        output_seq = []
        output_quality = np.zeros((rows, self.output_length))

        curr_row = 0

        for fastq in parsed_fastqs:
            sequence = fastq.sequence
            quality = fastq.phred_quality
            length = len(sequence)
            for i in range(length - self.k + 1):
                input_seq_vector = []
                output_seq_vector = []
                for j in range(self.input_length):
                    input_seq_vector.append(sequence[i+j])
                    input_quality[curr_row][j] = quality[i+j]

                output_offset = i + self.input_length + self.spacing
                for j in range(self.output_length):
                    output_seq_vector.append(sequence[output_offset + j])
                    output_quality[curr_row][j] = quality[output_offset + j]
                input_seq.append(input_seq_vector)
                output_seq.append(output_seq_vector)
                curr_row += 1

        return input_seq, input_quality, output_seq, output_quality
