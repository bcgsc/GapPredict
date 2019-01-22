import numpy as np
import matplotlib.pyplot as plt

class SequenceRegenerationViz:
    def compare_sequences(self, actual, predicted, seed_length):
        length_to_compare_to = min(len(actual), len(predicted))
        seed_length = min(seed_length, length_to_compare_to)

        comparison_string = ""

        for i in range(seed_length):
            comparison_string += "S"

        for i in range(length_to_compare_to - seed_length):
            idx = seed_length + i
            if actual[idx] == predicted[idx]:
                comparison_string += "-"
            else:
                comparison_string += "X"

        print("ACTUAL")
        print()
        print(actual)
        print(comparison_string)
        print(predicted)
        print()
        print("PREDICTED")

    def sliding_window_average_plot(self, correctness_vector, window_length=10):
        bases = len(correctness_vector)
        position = np.arange(bases) + 1
        averages = np.zeros(bases)
        for i in range(bases):
            if i < window_length:
                window = correctness_vector[0:i+window_length+1]
            else:
                window = correctness_vector[i-window_length:i+window_length+1]
            average = np.mean(window)
            averages[i] = average

        plt.figure()
        label = 'sliding_window_probability'
        plt.plot(position, averages)
        plt.savefig(label + '.png')
        plt.clf()

    def top_base_probability_plot(self, class_probabilities, correct_index_vector, top=2):
        position = np.arange(len(correct_index_vector)) + 1
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
            label = 'top_' + str(i+1) + '_probability'
            plt.scatter(position, y[i])
            plt.savefig(label + '.png')
            plt.clf()

        plt.figure()
        plt.scatter(position, correct_base_probability)
        plt.savefig('correct_base_probability.png')
        plt.clf()