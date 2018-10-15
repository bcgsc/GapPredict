class SequenceMatchCalculator:
    def __init__(self):
        pass

    def compare_sequences(self, seq1, seq2, start_idx=0, bases_to_check=None):
        #TODO: assumes equal length
        mismatches = 0
        length = len(seq1) - max(start_idx, 0)

        if bases_to_check != None:
            length = min(len(seq1) - start_idx, max(bases_to_check, 0))

        for i in range(length):
            idx = i + start_idx
            if seq1[idx] != seq2[idx]:
                mismatches += 1
        return mismatches




