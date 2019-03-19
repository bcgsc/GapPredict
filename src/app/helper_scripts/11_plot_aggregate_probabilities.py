import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import utils.directory_utils as dir_utils
from preprocess.KmerLabelEncoder import KmerLabelEncoder


def set_up_plot():
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    font = {
        'size': 25
    }
    plt.rc('font', **font)

    figure_dimensions=(12, 8)

    plt.figure(figsize=figure_dimensions)

def read_and_encode_actual_sequence(path, min_seed_length, line_offset):
    file = open(path)
    for i in range(line_offset):
        file.readline()
    seq = file.readline()
    seq = seq[min_seed_length:len(seq) - 1]

    encoder = KmerLabelEncoder()
    encoded_seq = encoder.encode_kmers([seq], [], with_shifted_output=False)[0][0]
    return encoded_seq

def dereference_indices(probability, index):
    assert len(probability) == len(index)
    array = np.zeros(len(index))

    for i in range(len(index)):
        array[i] = probability[i][index[i]]
    return array

def aggregate_probability(probability):
    return np.sum(np.log(probability))

def main():
    if os.name == 'nt':
        root = 'E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\src\\app\\new_rnn\\out\\models\\'
    else:
        root = '/home/echen/Desktop/Projects/Sealer_NN/src/app/new_rnn/out/models/'

    terminal_char = dir_utils.get_terminal_directory_character()

    output_folder = root + ".." + terminal_char + "aggregate" + terminal_char + "probability_aggregate" + terminal_char

    dir_utils.mkdir(output_folder)

    lstm_cell_directories = os.listdir(root)
    replicates = 6

    ids = set()
    for lstm_cells in lstm_cell_directories:
        dim_path = root + lstm_cells + terminal_char
        experiments = os.listdir(dim_path)
        for id in experiments:
            fasta_id = id.split("_R_")[0]
            ids.add(fasta_id)

    greedy_predicted = "greedy_predicted_probabilities.npy"
    teacher_forcing_predicted = "predicted_probabilities.npy"
    random_predicted = "random_predicted_probabilities.npy"

    headers = np.array(["", "type", "sum(ln(probability))"])
    numeric_headers = headers[2:]
    data_list = [headers]

    lf = "left_flank"
    rf = "right_flank"
    f = "forward"
    rc = "reverse_complement"

    flanks = [lf, rf]
    strands = [f, rc]

    row_id = 0
    for id in ids:
        assert(len(ids) == 1)
        for lstm_cells in lstm_cell_directories:
            tokens = lstm_cells.split("_")
            min_seed_length = int(tokens[2])

            for i in range(replicates):
                folder = id + "_R_" + str(i)
                folder_path = root + lstm_cells + terminal_char + folder + terminal_char + "regenerate_seq" + terminal_char

                for flank in flanks:
                    for strand in strands:
                        probability_folder = folder_path + flank + terminal_char + strand + terminal_char
                        greedy_probabilities = np.max(np.load(probability_folder + greedy_predicted), axis=1)

                        teacher_force_txt = probability_folder + "align.txt"
                        actual_encoded_sequence = read_and_encode_actual_sequence(teacher_force_txt, min_seed_length, 2)
                        teacher_forcing_probabilities_all = np.load(probability_folder + teacher_forcing_predicted)
                        teacher_forcing_probabilities = dereference_indices(teacher_forcing_probabilities_all, actual_encoded_sequence)

                        random_probabilities = np.load(probability_folder + random_predicted)

                        aggregate_greedy = aggregate_probability(greedy_probabilities)
                        aggregate_teacher_force = aggregate_probability(teacher_forcing_probabilities)
                        aggregate_random = aggregate_probability(random_probabilities)

                        data_list.append([row_id, "greedy", aggregate_greedy])
                        row_id += 1
                        data_list.append([row_id, "teacher_force", aggregate_teacher_force])
                        row_id += 1
                        data_list.append([row_id, "random", aggregate_random])
                        row_id += 1

    data = np.array(data_list)
    df = pd.DataFrame(data=data[1:, 1:], index=data[1:,0], columns=data[0, 1:])

    df[numeric_headers] = df[numeric_headers].apply(pd.to_numeric)

    set_up_plot()
    sns.boxplot(data=df, x="type", y="sum(ln(probability))")
    fig = plt.savefig(output_folder + "aggregate_probability.png")
    plt.close(fig)

if __name__ == "__main__":
    main()