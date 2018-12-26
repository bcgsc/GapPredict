import gzip
import mimetypes

from preprocess.SequenceParser import SequenceParser
from preprocess.SequenceReverser import SequenceReverser


class SequenceImporter:
    def __init__(self):
        self.parser = SequenceParser()
        self.reverser = SequenceReverser()

    def _open_file(self, path):
        file_extension = mimetypes.guess_type(path)
        if file_extension[1] == "gzip":
            file = gzip.open(path, 'rt')
        else:
            file = open(path, 'r')
        return file

    def import_fastq(self, paths, include_reverse_complement=False):
        reads = []

        buf = [None] * 4
        for path in paths:
            #TODO we could make a checker function to filter paths before going here (right now
            # we just eventually throw an exception)
            file = self._open_file(path)
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
