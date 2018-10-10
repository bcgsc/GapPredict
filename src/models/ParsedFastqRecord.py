import numpy as np


class ParsedFastqRecord:
    #TODO: strings suck in python, perhaps we should just make the sequence a list of characters from the get go,
    # this should help the SequenceReverser as well
    def __init__(self, sequence, phred_quality):
        self.sequence = sequence
        self.phred_quality = phred_quality

    def __eq__(self, other):
        return self.sequence == other.sequence and np.array_equal(self.phred_quality, other.phred_quality)