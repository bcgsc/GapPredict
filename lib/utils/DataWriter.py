import numpy as np
import utils.directory_utils as dir_utils
from preprocess.SequenceReverser import SequenceReverser

class DataWriter:
    def __init__(self, root_directory, directory=None):
        self.root_path = dir_utils.clean_directory_string(root_directory)

        if directory is not None:
            self.root_path = dir_utils.clean_directory_string(self.root_path + directory)

        dir_utils.mkdir(self.root_path)

    def _get_match_char(self, match):
        if match:
            return "-"
        else:
            return "X"

    def save_complements(self, forward_prediction, reverse_complement, fig_id, postfix="", fasta_ref=None):
        file_name = self.root_path + 'gap_predict_align.fa'
        file = open(file_name, 'w+')
        reverser = SequenceReverser()
        forward_pred = forward_prediction
        reverse_pred = reverser.reverse_complement(reverse_complement)

        file.write(">" + fig_id + "_forward" + postfix + "\n")
        file.write(forward_pred + '\n')
        file.write(">" + fig_id + "_reverse_complement" + postfix + "\n")
        file.write(reverse_pred + '\n')

        if fasta_ref is not None:
            with open(fasta_ref) as fasta:
                for line in fasta:
                    file.write(line)

        file.close()

    def _write_fasta(self, fig_id, seq, file):
        file.write('>' + fig_id + "\n")
        file.write(seq + "\n")

    def write_beam_search_results(self, predictions, flank_id):
        file = open(self.root_path + "beam_search_predictions.fasta", "w+")
        for i in range(len(predictions)):
            self._write_fasta(flank_id + "_" + str(i), predictions[i], file)
        file.close()

    def write_flank_predict_fasta(self, forward_left_flank, rc_left_flank, forward_right_flank, rc_right_flank,
                                  latent_dim, fig_id):
        file = open(self.root_path + 'flank_predict.fasta', 'w+')
        self._write_fasta(fig_id + "_left_flank_forward_LD_" + str(latent_dim), forward_left_flank, file)
        self._write_fasta(fig_id + "_left_flank_reverse_complement_LD_" + str(latent_dim), rc_left_flank, file)
        self._write_fasta(fig_id + "_right_flank_forward_LD_" + str(latent_dim), forward_right_flank, file)
        self._write_fasta(fig_id + "_right_flank_reverse_complement_LD_" + str(latent_dim), rc_right_flank, file)
        file.close()

    def save_probabilities(self, probabilities, file_id=None):
        if file_id is not None:
            id_string = file_id + "_"
        else:
            id_string = ""
        path = self.root_path + id_string + "predicted_probabilities"
        np.save(path, probabilities)