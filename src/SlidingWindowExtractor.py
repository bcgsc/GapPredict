from constants import EncodingConstants as CONSTANTS
from exceptions.SlidingWindowParamException import SlidingWindowParamException


class SlidingWindowExtractor:
    def __init__(self, input_length, spacing, output_length):
        if input_length <= 0 or output_length <= 0:
            raise SlidingWindowParamException("Lengths must be positive")
        if spacing < 0:
            raise SlidingWindowParamException("Spacing cannot be negative")
        self.input_length = input_length
        self.spacing = spacing
        self.output_length = output_length
        self.k = self.input_length + self.spacing + self.output_length

    def _validate_input_output(self, input, output):
        is_invalid = False
        for l in range(self.input_length):
            if input[l] not in CONSTANTS.BASES:
                is_invalid = True
                break

        for l in range(self.output_length):
            if output[l] not in CONSTANTS.BASES:
                is_invalid = True
                break

        return is_invalid


    def _extract_all_kmers_from_sequence(self, parsed_fastqs):
        input_kmers = []
        output_kmers = []
        quality_vectors = []
        for fastq in parsed_fastqs:
            sequence = fastq.sequence
            quality = fastq.phred_quality
            for i in range(len(sequence) - self.k + 1):
                input_seq = sequence[i:i + self.input_length]

                output_offset = i + self.input_length + self.spacing
                output_seq = sequence[output_offset:output_offset + self.output_length]

                quality_vector = quality[i:i + self.input_length]

                is_invalid = self._validate_input_output(input_seq, output_seq)

                if is_invalid:
                    continue

                input_kmers.append(input_seq)
                output_kmers.append(output_seq)
                quality_vectors.append(quality_vector)
        return input_kmers, output_kmers, quality_vectors

    def _extract_unique_kmers_from_sequence(self, parsed_fastqs):
        input_kmers = []
        output_kmers = []
        quality_vectors = []
        read_set = set()
        for fastq in parsed_fastqs:
            sequence = fastq.sequence
            quality = fastq.phred_quality
            for i in range(len(sequence) - self.k + 1):
                output_offset = i + self.input_length + self.spacing
                input_seq = sequence[i:i + self.input_length]
                output_seq = sequence[output_offset:output_offset + self.output_length]
                input_output = input_seq + output_seq
                if input_output not in read_set:
                    read_set.add(input_output)
                    quality_vector = quality[i:i + self.input_length]

                    is_invalid = self._validate_input_output(input_seq, output_seq)

                    if is_invalid:
                        continue

                    input_kmers.append(input_seq)
                    output_kmers.append(output_seq)
                    quality_vectors.append(quality_vector)
        return input_kmers, output_kmers, quality_vectors


    def extract_kmers_from_sequence(self, parsed_fastqs, unique=False):
        if unique:
            return self._extract_unique_kmers_from_sequence(parsed_fastqs)
        else:
            return self._extract_all_kmers_from_sequence(parsed_fastqs)