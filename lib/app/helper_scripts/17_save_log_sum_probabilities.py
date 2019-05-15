import os
import sys

if os.name == 'nt':
    sys.path.append('E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\lib\\')
else:
    sys.path.append('/home/echen/Desktop/Projects/Sealer_NN/lib/')

import numpy as np
import pandas as pd

import utils.directory_utils as dir_utils
from preprocess.KmerLabelEncoder import KmerLabelEncoder

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
    output_folder = "/home/echen/Desktop/Projects/Sealer_NN/lib/app/new_rnn/out/aggregate" + terminal_char + "gap_prediction_probability_aggregate" + terminal_char

    beam_search_probability = "beam_search_predicted_probabilities.npy"

    headers = np.array(["gap_id", "algorithm", "is_fixed", "is_left", "log-sum-probability"])
    numeric_headers = headers[2:]
    data_list = []
    predictions = ["forward", "reverse_complement"]  # forward = left, rc = right

    for gap_type in gap_types:
        root = '/projects/btl/scratch/echen/Apr_15_Backup/predict/' + gap_type + terminal_char
        dir_utils.mkdir(output_folder)

        gap_ids = os.listdir(root)
        num_replicates = 1

        ids = set()
        for gap_id in gap_ids:
            ids.add(gap_id)

        is_fixed = 1 if gap_type == "fixed" else 0

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
                    is_left = 1 if prediction == "forward" else 0
                    probability_folder = gap_prediction_folder + prediction + terminal_char
                    greedy_probabilities = np.max(np.load(probability_folder + "predicted_probabilities.npy"), axis=1)

                    aggregate_greedy = aggregate_probability(greedy_probabilities)

                    data_list.append([gap_id, "greedy", is_fixed, is_left, aggregate_greedy])

                beam_search_flank_prediction_folder = parent_folder_path + "beam_search" + terminal_char + "predict_gap" + terminal_char
                for prediction in predictions:
                    is_left = 1 if prediction == "forward" else 0
                    probability_folder = beam_search_flank_prediction_folder + prediction + terminal_char
                    lg_sum_probabilities = np.load(probability_folder + beam_search_probability)
                    sorted_sum = np.sort(lg_sum_probabilities)[::-1]
                    top_one = sorted_sum[0]

                    data_list.append([gap_id, "beam_search", is_fixed, is_left, top_one])

    data = np.array(data_list)
    df = pd.DataFrame(data=data, columns=headers)

    df[numeric_headers] = df[numeric_headers].apply(pd.to_numeric)
    df.to_csv(output_folder + "prediction_metrics.csv")


if __name__ == "__main__":
    main()