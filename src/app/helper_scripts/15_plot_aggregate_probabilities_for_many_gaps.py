import os
import sys

if os.name == 'nt':
    sys.path.append('E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\src\\')
else:
    sys.path.append('/home/echen/Desktop/Projects/Sealer_NN/src/')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import utils.directory_utils as dir_utils
from preprocess.KmerLabelEncoder import KmerLabelEncoder

primary_text_font_size=45
secondary_text_font_size=30
linewidth=6
rotation=75

def save_plot(data, path):
    plt.rc('xtick', labelsize=secondary_text_font_size)
    plt.rc('ytick', labelsize=secondary_text_font_size)
    font = {
        'size': primary_text_font_size
    }
    plt.rc('font', **font)

    figure_dimensions=(13, 16)

    plt.figure(figsize=figure_dimensions)

    flierprops = {
        'markersize': 10,
        'markerfacecolor': 'red',
        'marker': 'o'
    }

    ax = sns.boxplot(data=data, x="algorithm", y="log-sum-probability", linewidth=linewidth, flierprops=flierprops)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=rotation)
    plt.tight_layout()
    fig = plt.savefig(path)
    plt.close(fig)

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
    terminal_char = dir_utils.get_terminal_directory_character()
    gap_types = ["fixed", "unfixed"]

    for gap_type in gap_types:
        root = '/projects/btl/scratch/echen/April_2_Results_Backup/scratch/' + gap_type + terminal_char
        output_folder = "/home/echen/Desktop/Projects/Sealer_NN/src/app/new_rnn/out/aggregate" + terminal_char + "gap_prediction_probability_aggregate" + terminal_char + gap_type + terminal_char
        dir_utils.mkdir(output_folder)

        gap_ids = os.listdir(root)
        num_replicates = 1

        ids = set()
        for gap_id in gap_ids:
            ids.add(gap_id)

        greedy_predicted = "greedy_predicted_probabilities.npy"
        teacher_forcing_predicted = "predicted_probabilities.npy"
        random_predicted = "random_predicted_probabilities.npy"
        beam_search_probability = "beam_search_predicted_probabilities.npy"

        headers = np.array(["", "algorithm", "log-sum-probability"])
        numeric_headers = headers[2:]
        data_list = [headers]

        lf = "left_flank"
        rf = "right_flank"
        f = "forward"
        rc = "reverse_complement"

        flanks = [lf, rf]
        strands = [f, rc]

        row_id = 0
        min_seed_length=52
        for gap_id in ids:
            gap_folder = root + gap_id + terminal_char
            for i in range(num_replicates):
                model_folder = gap_id + "_R_" + str(i)
                parent_folder_path = gap_folder + model_folder + terminal_char
                files = os.listdir(parent_folder_path)
                if len(files) == 1:
                    continue

                flank_prediction_folder = parent_folder_path + "regenerate_seq" + terminal_char

                for flank in flanks:
                    for strand in strands:
                        probability_folder = flank_prediction_folder + flank + terminal_char + strand + terminal_char
                        greedy_probabilities = np.max(np.load(probability_folder + greedy_predicted), axis=1)

                        teacher_force_txt = probability_folder + "align.txt"
                        actual_encoded_sequence = read_and_encode_actual_sequence(teacher_force_txt, min_seed_length, 2)
                        teacher_forcing_probabilities_all = np.load(probability_folder + teacher_forcing_predicted)
                        teacher_forcing_probabilities = dereference_indices(teacher_forcing_probabilities_all,
                                                                            actual_encoded_sequence)

                        random_probabilities = np.load(probability_folder + random_predicted)

                        aggregate_greedy = aggregate_probability(greedy_probabilities)
                        aggregate_teacher_force = aggregate_probability(teacher_forcing_probabilities)
                        aggregate_random = aggregate_probability(random_probabilities)

                        data_list.append([row_id, "greedy", aggregate_greedy])
                        row_id += 1
                        data_list.append([row_id, "teachforce", aggregate_teacher_force])
                        row_id += 1
                        data_list.append([row_id, "random", aggregate_random])
                        row_id += 1

                beam_search_flank_prediction_folder = parent_folder_path + "beam_search" + terminal_char + "regenerate_seq" + terminal_char
                for flank in flanks:
                    for strand in strands:
                        probability_folder = beam_search_flank_prediction_folder + flank + terminal_char + strand + terminal_char
                        lg_sum_probabilities = np.load(probability_folder + beam_search_probability)
                        sorted_sum = np.sort(lg_sum_probabilities)[::-1]
                        top_four = sorted_sum[0:4]
                        top_one = sorted_sum[0]

                        for sum in sorted_sum:
                            data_list.append([row_id, "bmsrch_all", sum])
                            row_id += 1

                        for sum in top_four:
                            data_list.append([row_id, "bmsrch_4", sum])
                            row_id += 1

                        data_list.append([row_id, "bmsrch_1", top_one])
                        row_id += 1

        data = np.array(data_list)
        df = pd.DataFrame(data=data[1:, 1:], index=data[1:,0], columns=data[0, 1:])

        df[numeric_headers] = df[numeric_headers].apply(pd.to_numeric)
        path = output_folder + "flanks_aggregate_probability.png"
        save_plot(df, path)


        df_drilldown = df.loc[df['algorithm'] != "random"]
        path = output_folder + "flanks_aggregate_probability_drilldown.png"
        save_plot(df_drilldown, path)

        ####################################################################################################################
        ####GAPS######
        ####################################################################################################################

        headers = np.array(["", "algorithm", "log-sum-probability"])
        numeric_headers = headers[2:]
        data_list = [headers]
        predictions = ["forward", "reverse_complement"] #forward = left, rc = right

        row_id = 0
        for gap_id in ids:
            gap_folder = root + gap_id + terminal_char
            for i in range(num_replicates):
                model_folder = gap_id + "_R_" + str(i)
                parent_folder_path = gap_folder + model_folder + terminal_char
                files = os.listdir(parent_folder_path)
                if len(files) == 1:
                    continue

                gap_prediction_folder = parent_folder_path + "predict_gap" + terminal_char

                for prediction in predictions:
                    probability_folder = gap_prediction_folder + prediction + terminal_char
                    greedy_probabilities = np.max(np.load(probability_folder + "predicted_probabilities.npy"), axis=1)

                    aggregate_greedy = aggregate_probability(greedy_probabilities)

                    data_list.append([row_id, "greedy", aggregate_greedy])
                    row_id += 1

                beam_search_flank_prediction_folder = parent_folder_path + "beam_search" + terminal_char + "predict_gap" + terminal_char
                for prediction in predictions:
                    probability_folder = beam_search_flank_prediction_folder + prediction + terminal_char
                    lg_sum_probabilities = np.load(probability_folder + beam_search_probability)
                    sorted_sum = np.sort(lg_sum_probabilities)[::-1]
                    top_four = sorted_sum[0:4]
                    top_one = sorted_sum[0]

                    for sum in sorted_sum:
                        data_list.append([row_id, "bmsrch_all", sum])
                        row_id += 1

                    for sum in top_four:
                        data_list.append([row_id, "bmsrch_4", sum])
                        row_id += 1

                    data_list.append([row_id, "bmsrch_1", top_one])
                    row_id += 1

        data = np.array(data_list)
        df = pd.DataFrame(data=data[1:, 1:], index=data[1:,0], columns=data[0, 1:])

        df[numeric_headers] = df[numeric_headers].apply(pd.to_numeric)
        path = output_folder + "gaps_aggregate_probability.png"
        save_plot(df, path)

if __name__ == "__main__":
    main()