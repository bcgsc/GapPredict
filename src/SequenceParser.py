import numpy as np

from BaseQualityConverter import BaseQualityConverter


class ParsedFastqRecord:
    def __init__(self, sequence, phred_quality):
        self.sequence = sequence
        self.phred_quality = phred_quality

    def __eq__(self, other):
        return self.sequence == other.sequence and np.array_equal(self.phred_quality, other.phred_quality)

class SequenceParser:
    def __init__(self):
        self.base_quality_convert = BaseQualityConverter()
        return

    def strip_terminal_newline(self, string):
        if(string[len(string) - 1] == "\n"):
            return string[0:len(string)-1]
        else:
            return string

    def parse_fastq(self, id, sequence, optional_id, quality_string):
        filtered_sequence = self.strip_terminal_newline(sequence)
        filtered_quality = self.strip_terminal_newline(quality_string)
        phred_quality = self.base_quality_convert.convert_quality_to_phred(filtered_quality)
        return ParsedFastqRecord(filtered_sequence, phred_quality)


