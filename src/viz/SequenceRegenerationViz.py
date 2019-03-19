import os

import matplotlib.pyplot as plt
import numpy as np

import utils.directory_utils as dir_utils
from preprocess.SequenceComplementCalculator import SequenceComplementCalculator
from preprocess.SequenceReverser import SequenceReverser


class SequenceRegenerationViz:
    def __init__(self, root_directory=None, directory=None):
        if root_directory is None:
            if os.name == 'nt':
                self.root_path = 'E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\src\\app\\new_rnn\\out\\predict_results\\'
            else:
                self.root_path = '/home/echen/Desktop/Projects/Sealer_NN/src/app/new_rnn/out/predict_results/'
        else:
            self.root_path = dir_utils.clean_directory_string(root_directory)

        if directory is not None:
            self.root_path = dir_utils.clean_directory_string(self.root_path + directory)

        dir_utils.mkdir(self.root_path)

    def _configure_plot(self):
        figure_dimensions = (12, 9)
        plt.figure(figsize=figure_dimensions)
        plt.rc('xtick', labelsize=14)
        plt.rc('ytick', labelsize=14)
        font = {
            'size': 16
        }
        plt.rc('font', **font)

    def _get_match_char(self, match):
        if match:
            return "-"
        else:
            return "X"

    def align_complements(self, forward_prediction, reverse_complement, seed_length, static_offset=0):
        complement_validator = SequenceComplementCalculator()
        file = open(self.root_path + 'bidirectional_align.txt', 'w+')
        padded_forward_pred = "N" * static_offset + forward_prediction
        padded_reverse_pred = ("N" * static_offset + reverse_complement)[::-1]
        matches = complement_validator.compare_sequences(padded_forward_pred, padded_reverse_pred)

        forward_comparison_string = ""
        reverse_comparison_string = ""

        for i in range(static_offset):
            forward_comparison_string += self._get_match_char(matches[i])

        for i in range(seed_length):
            forward_comparison_string += "S"

        for i in range(len(forward_prediction) - seed_length):
            forward_comparison_string += self._get_match_char(matches[i + static_offset + seed_length])

        for i in range(len(reverse_complement) - seed_length):
            reverse_comparison_string += self._get_match_char(matches[i])

        for i in range(seed_length):
            reverse_comparison_string += "S"

        for i in range(static_offset):
            reverse_comparison_string += self._get_match_char(matches[i + len(reverse_complement) + seed_length])

        file.write("FORWARD\n")
        file.write('\n')
        file.write(padded_forward_pred + '\n')
        file.write(forward_comparison_string + '\n')
        file.write(reverse_comparison_string + '\n')
        file.write(padded_reverse_pred + '\n')
        file.write('\n')
        file.write('REVERSE COMPLEMENT\n')
        meaningful_matches = matches[static_offset:len(matches)-static_offset]
        file.write('Mean Match: ' + str(round(np.mean(meaningful_matches), 2)) +'\n')
        file.close()

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
        file = open(self.root_path + "predict.fasta", "w+")
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

    def compare_sequences(self, actual, predicted, seed_length, matches, offset=0, append=False, fig_id=None):
        static_padding = ""

        for i in range(offset):
            static_padding += " "

        comparison_string = str(static_padding)
        for i in range(seed_length):
            comparison_string += "S"

        for i in range(len(matches)):
            comparison_string += self._get_match_char(matches[i])

        first_mismatch_idx = None
        for i in range(len(matches)):
            if not matches[i]:
                first_mismatch_idx = offset + seed_length + i
                break

        mode = "a" if append else "w+"

        if fig_id is not None:
            id_string = fig_id + "_"
        else:
            id_string = ""
        file = open(self.root_path + id_string + 'align.txt', mode)

        file.write("ACTUAL\n")
        file.write('\n')
        file.write(actual + '\n')
        file.write(comparison_string + '\n')
        file.write(static_padding + predicted + '\n')
        file.write('\n')
        file.write('PREDICTED\n')
        file.write('Mean Match: ' + str(round(np.mean(matches), 2)) +'\n')
        if first_mismatch_idx is not None:
            file.write('First Mismatch: ' + str(first_mismatch_idx) + '\n')
        file.write('\n')
        file.write('-----------------------------------------------------\n')
        file.write('\n')
        file.close()

    def sliding_window_average(self, vector, window_length=10):
        length = len(vector)
        averages = np.zeros(length)
        for i in range(length):
            if i < window_length:
                window = vector[0:i+window_length+1]
            else:
                window = vector[i-window_length:i+window_length+1]
            average = np.mean(window)
            averages[i] = average
        return averages

    def sliding_window_average_plot(self, correctness_vector, window_length=10, offset=0, fig_id=""):
        bases = len(correctness_vector)
        position = np.arange(bases) + 1 + offset
        averages = self.sliding_window_average(correctness_vector, window_length=window_length)

        self._configure_plot()
        plt.plot(position, averages)
        plt.ylim(0, 1.1)
        plt.xlim(0, int(max(position) * 1.05))
        plt.xlabel("Base Index")
        plt.ylabel("Avg Probability (Window = " + str(window_length) + ")")
        fig = plt.savefig(self.root_path + fig_id + 'sliding_window_probability.png')
        plt.close(fig)

    def save_probabilities(self, probabilities, fig_id=None):
        if fig_id is not None:
            id_string = fig_id + "_"
        else:
            id_string = ""
        path = self.root_path + id_string + "predicted_probabilities"
        np.save(path, probabilities)

    def top_base_probability_plot(self, class_probabilities, correct_index_vector, top=2, offset=0, fig_id=""):
        position = np.arange(len(correct_index_vector)) + 1 + offset
        sorted_probabilities = np.sort(class_probabilities, axis=1)

        y = []

        for i in range(top):
            col = sorted_probabilities[:, len(sorted_probabilities[0]) - i - 1]
            y.append(col)

        correct_base_probability = np.zeros(len(correct_index_vector))
        for i in range(len(correct_index_vector)):
            idx = correct_index_vector[i]
            probability = class_probabilities[i][idx]
            correct_base_probability[i] = probability

        for i in range(top):
            self._configure_plot()
            label = self.root_path + fig_id + 'top_' + str(i + 1) + '_probability'
            plt.scatter(position, y[i])
            plt.ylim(0, 1.1)
            plt.xlim(0, int(max(position) * 1.05))
            plt.xlabel("Base Index")
            plt.ylabel("Probability")
            fig = plt.savefig(label + '.png')
            plt.close(fig)

        self._configure_plot()
        plt.scatter(position, correct_base_probability)
        plt.ylim(0, 1.1)
        plt.xlim(0, int(max(position) * 1.05))
        plt.xlabel("Base Index")
        plt.ylabel("Probability")
        fig = plt.savefig(self.root_path + fig_id + 'correct_base_probability.png')
        plt.close(fig)