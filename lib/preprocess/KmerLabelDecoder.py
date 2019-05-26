import constants.EncodingConstants as CONSTANTS

class KmerLabelDecoder:
    def __init__(self):
        pass

    def decode(self, integer_encoding):
        string = "".join(CONSTANTS.REVERSE_INTEGER_ENCODING[integer_encoding.astype(int)])
        return string