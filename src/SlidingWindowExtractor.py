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

    def extract_kmers_from_sequence(self, parsed_fastqs):
        input_kmers = []
        output_kmers = []
        quality_vectors = []
        BASES = CONSTANTS.BASES
        for fastq in parsed_fastqs:
            sequence = fastq.sequence
            quality = fastq.phred_quality
            length = len(sequence)
            for i in range(length - self.k + 1):
                input_seq = sequence[i:i+self.input_length]

                output_offset = i + self.input_length + self.spacing
                output_seq = sequence[output_offset:output_offset+self.output_length]

                quality_vector = quality[i:i+self.input_length]

                skip_kmer = False
                for l in range(self.input_length):
                    if input_seq[l] not in BASES:
                        skip_kmer = True
                        break

                for l in range(self.output_length):
                    if output_seq[l] not in BASES:
                        skip_kmer = True
                        break

                if skip_kmer:
                    continue

                input_kmers.append(input_seq)
                output_kmers.append(output_seq)
                quality_vectors.append(quality_vector)
        return input_kmers, output_kmers, quality_vectors