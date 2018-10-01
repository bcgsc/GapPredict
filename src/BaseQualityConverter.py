import numpy as np


class BaseQualityConverter:
    def __init__(self):
        pass

    def convert_quality_to_phred(self, quality_string):
        seq_length = len(quality_string)
        phred_quality = np.zeros(seq_length)

        for i in range(seq_length):
            qual_char = quality_string[i]
            phred_quality[i] = ord(qual_char) - 33

        return phred_quality
