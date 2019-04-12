from models.ParsedFastaRecord import ParsedFastaRecord
from models.ParsedFastqRecord import ParsedFastqRecord
from preprocess.BaseQualityConverter import BaseQualityConverter

class SequenceParser:
    def __init__(self):
        pass

    def _strip_terminal_newline(self, string):
        if string[len(string) - 1] == "\n":
            return string[0:len(string)-1]
        else:
            return string

class RawSequenceParser(SequenceParser):
    def __init__(self):
        pass

    def parse_fastq(self, id, sequence, optional_id, quality_string):
        return super()._strip_terminal_newline(sequence).upper()

    def parse_fasta(self, buf):
        sequence = ""
        for subsequence in buf[1:]:
            sequence += super()._strip_terminal_newline(subsequence)
        return sequence.upper()



class FormattedSequenceParser(SequenceParser):
    def __init__(self):
        self.base_quality_convert = BaseQualityConverter()

    def parse_fastq(self, id, sequence, optional_id, quality_string):
        filtered_sequence = super()._strip_terminal_newline(sequence)
        filtered_quality = super()._strip_terminal_newline(quality_string)
        phred_quality = self.base_quality_convert.convert_quality_to_phred(filtered_quality)
        return ParsedFastqRecord(filtered_sequence, phred_quality)

    def parse_fasta(self, buf):
        filtered_id = super()._strip_terminal_newline(buf[0])
        sequence = ""
        for subsequence in buf[1:]:
            sequence += super()._strip_terminal_newline(subsequence)
        return ParsedFastaRecord(filtered_id, sequence)
