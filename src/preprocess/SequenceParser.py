from models.ParsedFastaRecord import ParsedFastaRecord
from models.ParsedFastqRecord import ParsedFastqRecord
from preprocess.BaseQualityConverter import BaseQualityConverter


class SequenceParser:
    def __init__(self):
        self.base_quality_convert = BaseQualityConverter()

    def _strip_terminal_newline(self, string):
        if string[len(string) - 1] == "\n":
            return string[0:len(string)-1]
        else:
            return string

    def parse_fastq(self, id, sequence, optional_id, quality_string):
        filtered_sequence = self._strip_terminal_newline(sequence)
        filtered_quality = self._strip_terminal_newline(quality_string)
        phred_quality = self.base_quality_convert.convert_quality_to_phred(filtered_quality)
        return ParsedFastqRecord(filtered_sequence, phred_quality)

    def parse_fasta(self, buf):
        filtered_id = self._strip_terminal_newline(buf[0])
        sequence = ""
        for subsequence in buf[1:]:
            sequence += self._strip_terminal_newline(subsequence)
        return ParsedFastaRecord(filtered_id, sequence)
