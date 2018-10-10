import gzip
import mimetypes

from SequenceParser import SequenceParser
from SequenceReverser import SequenceReverser


class SequenceImporter:
    def __init__(self):
        self.parser = SequenceParser()
        self.reverser = SequenceReverser()

    def import_fastq(self, path, include_reverse_complement=False):
        reads = []
        file_extension = mimetypes.guess_type(path)
        if file_extension[1] == "gzip":
            file = gzip.open(path, 'rt')
        else:
            file = open(path, 'r')

        buf = [None] * 4
        line_num = 0
        line = file.readline()
        while line:
            buf[line_num] = line
            if line_num == 3:
                parsed_fastq = self.parser.parse_fastq(buf[0], buf[1], buf[2], buf[3])
                reads.append(parsed_fastq)
                if include_reverse_complement:
                    reverse_complement = self.reverser.reverse_sequence(parsed_fastq)
                    reads.append(reverse_complement)

            line = file.readline()
            line_num = (line_num + 1) % 4
        file.close()
        return reads
