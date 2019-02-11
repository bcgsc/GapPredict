import matplotlib.pyplot as plt
import numpy as np
import os

if os.name == 'nt':
    root_path = 'E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\src\\app\\new_rnn\\out\\'
else:
    root_path = '/home/echen/Desktop/Projects/Sealer_NN/src/app/new_rnn/out/'

class SequenceRegenerationViz:
    def compare_multiple_sequences(self, reference_sequence, alignment_data, seed_length, static_offset=0):
        file = open(root_path + 'multi_align.txt', 'w+')

        for i in range(len(alignment_data)):
            alignment_tuple = alignment_data[i]
            predicted_sequence = alignment_tuple[0]
            matches = alignment_tuple[1]
            current_lower_bound = alignment_tuple[2]

            static_padding = ""
            for j in range(static_offset):
                static_padding += " "

            comparison_string = str(static_padding)
            for j in range(seed_length):
                comparison_string += "S"

            for j in range(len(matches)):
                if matches[j]:
                    comparison_string += "-"
                else:
                    comparison_string += "X"

            first_mismatch_idx = None
            for j in range(len(matches)):
                if not matches[j]:
                    first_mismatch_idx = seed_length + current_lower_bound + j
                    break

            padding = ""
            for j in range(current_lower_bound):
                padding += " "

            file.write("SEQ " + str(i) + "\n")
            file.write(reference_sequence + '\n')
            file.write(padding + comparison_string + '\n')
            file.write(static_padding + padding + predicted_sequence + '\n')
            file.write('\n')
            file.write('Mean Match: ' + str(round(np.mean(matches),2)) + '\n')
            if first_mismatch_idx is not None:
                file.write('First Mismatch: ' + str(first_mismatch_idx) + '\n')
            file.write('\n')
        file.close()


    def compare_sequences(self, actual, predicted, seed_length, matches, offset=0):
        static_padding = ""

        for i in range(offset):
            static_padding += " "

        comparison_string = str(static_padding)
        for i in range(seed_length):
            comparison_string += "S"

        for i in range(len(matches)):
            if matches[i]:
                comparison_string += "-"
            else:
                comparison_string += "X"

        first_mismatch_idx = None
        for i in range(len(matches)):
            if not matches[i]:
                first_mismatch_idx = seed_length + i
                break


        file = open(root_path + 'align.txt', 'w+')

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
        file.close()

    def sliding_window_average_plot(self, correctness_vector, window_length=10, offset=0, id=""):
        bases = len(correctness_vector)
        position = np.arange(bases) + 1 + offset
        averages = np.zeros(bases)
        for i in range(bases):
            if i < window_length:
                window = correctness_vector[0:i+window_length+1]
            else:
                window = correctness_vector[i-window_length:i+window_length+1]
            average = np.mean(window)
            averages[i] = average

        plt.figure()
        plt.plot(position, averages)
        plt.ylim(0, 1.1)
        plt.xlim(0, int(max(position) * 1.05))
        plt.xlabel("Base Index")
        plt.ylabel("Avg Probability (Window = " + str(window_length) + ")")
        plt.savefig(root_path + id + 'sliding_window_probability.png')
        plt.clf()

    def top_base_probability_plot(self, class_probabilities, correct_index_vector, top=2, offset=0, id=""):
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
            plt.figure()
            label = root_path + id + 'top_' + str(i+1) + '_probability'
            plt.scatter(position, y[i])
            plt.ylim(0, 1.1)
            plt.xlim(0, int(max(position) * 1.05))
            plt.xlabel("Base Index")
            plt.ylabel("Probability")
            plt.savefig(label + '.png')
            plt.clf()

        plt.figure()
        plt.scatter(position, correct_base_probability)
        plt.ylim(0, 1.1)
        plt.xlim(0, int(max(position) * 1.05))
        plt.xlabel("Base Index")
        plt.ylabel("Probability")
        plt.savefig(root_path + id + 'correct_base_probability.png')
        plt.clf()