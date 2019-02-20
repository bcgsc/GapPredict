import numpy as np

class SequenceMatchCalculator:
    def __init__(self):
        pass

    def compare_sequences(self, seq_matrix1, seq_matrix2, start_idx=0, bases_to_check=None):
        #TODO: assumes matrices have equal dimensions
        if len(seq_matrix1) == 0:
            return np.array([[]])
        num_seq = len(seq_matrix1)
        seq_length = len(seq_matrix1[0])
        fixed_start_idx = max(start_idx, 0)

        if bases_to_check != None:
            fixed_bases_to_check = max(bases_to_check, 0)
            seq_length = min(max(seq_length - fixed_start_idx, 0), fixed_bases_to_check)
        else:
            seq_length = max(seq_length - fixed_start_idx, 0)

        match_matrix = np.zeros((num_seq, seq_length))

        for i in range(num_seq):
            seq1 = seq_matrix1[i]
            seq2 = seq_matrix2[i]
            if len(seq1) == 0 or len(seq2) == 0:
                continue
            for j in range(seq_length):
                base_idx = j + start_idx
                if seq1[base_idx] == seq2[base_idx] or seq1[base_idx] == "N" or seq2[base_idx] == "N":
                    match_matrix[i][j] = 1

        return match_matrix




