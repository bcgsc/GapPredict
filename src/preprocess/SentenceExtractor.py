from constants import RnnEncodingConstants as CONSTANTS
from exceptions.NonpositiveLengthException import NonpositiveLengthException

class SentenceExtractor:
    def __init__(self):
        pass

    def _invalid(self, sequence):
        is_invalid = False
        for l in range(len(sequence)):
            if sequence[l] not in CONSTANTS.BASES:
                is_invalid = True
                break

        return is_invalid

    def split_sequences_into_kmers(self, sequences, k):
        if k < 1:
            raise NonpositiveLengthException

        sentences = []
        for fastq_read in sequences:
            sentence = []
            sequence = fastq_read.sequence
            if not self._invalid(sequence):
                for i in range(len(sequence) - k + 1):
                    sentence.append(sequence[i:i+k])
                if len(sentence) == 0:
                    sentence.append(sequence)
                sentences.append(sentence)
        return sentences