from constants.PreprocessConstants import COMPLEMENT_MAP as COMPLEMENT_MAP
from models.ParsedFastqRecord import ParsedFastqRecord


class SequenceReverser:
    def __init__(self):
        pass

    def reverse_complement(self, sequence):
        reverse_seq = ""
        for i in range(len(sequence)):
            reverse_idx = len(sequence) - i - 1
            reverse_seq += COMPLEMENT_MAP[sequence[reverse_idx]]
        return reverse_seq

    def reverse_sequence(self, parsed_fastq):
        reverse_complement = self.reverse_complement(parsed_fastq.sequence)
        reverse_quality = parsed_fastq.phred_quality[::-1]
        return ParsedFastqRecord(reverse_complement, reverse_quality)