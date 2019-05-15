import math
import os
import sys

if os.name == 'nt':
    sys.path.append('E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\lib\\')
else:
    sys.path.append('/home/echen/Desktop/Projects/Sealer_NN/lib/')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import utils.directory_utils as dir_utils

primary_text_font_size=45
secondary_text_font_size=35
linewidth=5

def set_up_plot():
    plt.rc('xtick', labelsize=secondary_text_font_size)
    plt.rc('ytick', labelsize=secondary_text_font_size)
    font = {
        'size': primary_text_font_size
    }
    plt.rc('font', **font)

    figure_dimensions=(13, 13)

    plt.figure(figsize=figure_dimensions)

def save_fig(data, y_label, output_folder, file_name):
    set_up_plot()
    flierprops = {
        'markersize': 10,
        'markerfacecolor': 'red',
        'marker': 'o'
    }
    ax = sns.boxplot(data=data, x="seed_range", y=y_label, hue="lstm_cells", linewidth=linewidth, flierprops=flierprops)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=60)
    plt.tight_layout()

    legend_font = {
        'size': secondary_text_font_size
    }
    plt.legend(loc="best", prop=legend_font)
    fig = plt.savefig(output_folder + file_name)
    plt.close(fig)


def main():
    if os.name == 'nt':
        root = 'E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\lib\\app\\new_rnn\\out\\models\\'
    else:
        root = '/home/echen/Desktop/Projects/Sealer_NN/lib/app/new_rnn/out/models/'

    terminal_char = dir_utils.get_terminal_directory_character()

    output_folder = root + ".." + terminal_char + "aggregate" + terminal_char + "training_aggregate" + terminal_char

    dir_utils.mkdir(output_folder)

    root = '/projects/btl/scratch/echen/March_26_Results_Backup/models/'
    lstm_cell_directories = os.listdir(root)
    replicates = 30

    ids = set()
    for lstm_cells in lstm_cell_directories:
        dim_path = root + lstm_cells + terminal_char
        experiments = os.listdir(dim_path)
        for id in experiments:
            fasta_id = id.split("_R_")[0]
            ids.add(fasta_id)

    training_accuracy_file = "training_accuracy.npy"
    validation_accuracy_file = "validation_accuracy.npy"
    training_loss_file = "training_loss.npy"
    validation_loss_file = "validation_loss.npy"
    lengths_file = "lengths.npy"

    headers = np.array(["", "lstm_cells", "seed_range", "best_epoch", "valid_acc", "train_acc", "valid_loss", "train_loss",  "min_range", "max_range"])
    numeric_headers = headers[3:]
    data_list = [headers]

    ranges_to_filter = ["26-26"]
    range_set = set()
    for range_string in ranges_to_filter:
        range_set.add(range_string)

    row_id = 0
    for id in ids:
        assert(len(ids) == 1)
        for lstm_cells in lstm_cell_directories:
            for i in range(replicates):
                folder = id + "_R_" + str(i)
                folder_path = root + lstm_cells + terminal_char + folder + terminal_char
                model_files = os.listdir(folder_path)

                tokens = lstm_cells.split("_")
                dims = tokens[0]
                min_seed_length = tokens[2]
                max_seed_length = tokens[4]

                if int(max_seed_length) < int(min_seed_length):
                    max_seed_length = "N"

                full_range = min_seed_length + "-" + max_seed_length

                if full_range in range_set:
                    continue

                for results_file in model_files:
                    if results_file.startswith("BS"):
                        training_folder = folder_path + results_file + terminal_char
                        validation_accuracy = np.load(training_folder + validation_accuracy_file)
                        validation_loss = np.load(training_folder + validation_loss_file)

                        training_accuracy = np.load(training_folder + training_accuracy_file)
                        training_loss = np.load(training_folder + training_loss_file)
                        lengths = np.load(training_folder + lengths_file)

                        weighted_mean_valid_accuracy = np.sum(validation_accuracy * lengths, axis=1) / np.sum(lengths)
                        weighted_mean_valid_loss = np.sum(validation_loss * lengths, axis=1) / np.sum(lengths)

                        best_epoch = np.argmin(weighted_mean_valid_loss)
                        chosen_valid_accuracy = weighted_mean_valid_accuracy[best_epoch]
                        chosen_valid_loss = weighted_mean_valid_loss[best_epoch]
                        chosen_train_accuracy = training_accuracy[best_epoch]
                        chosen_train_loss = training_loss[best_epoch]

                        row = [row_id, dims, full_range, best_epoch, chosen_valid_accuracy,
                               chosen_train_accuracy, chosen_valid_loss, chosen_train_loss, int(min_seed_length), int(max_seed_length) if max_seed_length != "N" else math.inf]

                        data_list.append(row)
                        row_id += 1

    data = np.array(data_list)
    df = pd.DataFrame(data=data[1:, 1:], index=data[1:,0], columns=data[0, 1:])

    df[numeric_headers] = df[numeric_headers].apply(pd.to_numeric)
    df.sort_values(by=['min_range', 'max_range'], inplace=True)

    save_fig(df, "valid_acc", output_folder, "valid_acc_drilldown.png")
    save_fig(df, "train_acc", output_folder, "train_acc_drilldown.png")
    save_fig(df, "valid_loss", output_folder, "valid_loss_drilldown.png")
    save_fig(df, "train_loss", output_folder, "train_loss_drilldown.png")

if __name__ == "__main__":
    main()