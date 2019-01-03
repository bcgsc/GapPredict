import gzip
import mimetypes

class Importer:
    def __init__(self):
        pass

    def _open_file(self, path):
        file_extension = mimetypes.guess_type(path)
        if file_extension[1] == "gzip":
            file = gzip.open(path, 'rt')
        else:
            file = open(path, 'r')
        return file