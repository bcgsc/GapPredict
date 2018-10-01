class SequenceMatchCalculator:
    def __init__(self):
        pass

    def compare_sequences(self, seq1, seq2):
        #TODO: assumes equal length
        mismatches = 0
        for i in range(len(seq1)):
            if seq1[i] != seq2[i]:
                mismatches += 1
        return mismatches




