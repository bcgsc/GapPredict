import mimetypes
import gzip
from SequenceParser import SequenceParser

class SequenceImporter:
    def __init__(self):
        self.parser = SequenceParser()
        return

    def import_fastq(self, path):
        reads = []
        type = mimetypes.guess_type(path)
        file = None
        if(type[1] == "gzip"):
            file = gzip.open(path, 'rt')
        else:
            file = open(path, 'r')

        buf = [None] * 4
        line_num = 0
        line = file.readline()
        while line:
            buf[line_num] = line
            if(line_num == 3):
                reads.append(self.parser.parse_fastq(buf[0], buf[1], buf[2], buf[3]))

            line = file.readline()
            line_num = (line_num + 1) % 4

        return reads
