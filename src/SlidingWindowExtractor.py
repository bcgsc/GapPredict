from SequenceParser import ParsedFastqRecord


class SlidingWindowParamException(Exception):
    pass


class SlidingWindowExtractor:
    def __init__(self, input_length, spacing, output_length):
        if input_length <= 0 or spacing <= 0 or output_length <= 0:
            raise SlidingWindowParamException("Lengths must be positive")

        self.input_length = input_length
        self.spacing = spacing
        self.output_length = output_length

    def extract_input_output_from_sequence(self, parsed_fastq):
        # TODO: assumes that the sequence and quality are the same length
        sequence = parsed_fastq.sequence
        quality = parsed_fastq.phred_quality
        length = len(sequence)
        k = self.input_length + self.spacing + self.output_length

        input_seq = []
        output_seq = []
        for i in range(length - k + 1):
            input_seq.append(ParsedFastqRecord(sequence[i:i + self.input_length],
                                               quality[i:i + self.input_length]))
            output_seq.append(ParsedFastqRecord(sequence[i + self.input_length + self.spacing:i + k],
                                                quality[i + self.input_length + self.spacing:i + k]))

        return input_seq, output_seq
