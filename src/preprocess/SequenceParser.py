from preprocess.BaseQualityConverter import BaseQualityConverter
from models.ParsedFastqRecord import ParsedFastqRecord


class SequenceParser:
    def __init__(self):
        self.base_quality_convert = BaseQualityConverter()
        return

    def strip_terminal_newline(self, string):
        if string[len(string) - 1] == "\n":
            return string[0:len(string)-1]
        else:
            return string

    def parse_fastq(self, id, sequence, optional_id, quality_string):
        filtered_sequence = self.strip_terminal_newline(sequence)
        filtered_quality = self.strip_terminal_newline(quality_string)
        phred_quality = self.base_quality_convert.convert_quality_to_phred(filtered_quality)
        return ParsedFastqRecord(filtered_sequence, phred_quality)


