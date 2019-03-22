import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import utils.directory_utils as dir_utils


def set_up_plot():
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    font = {
        'size': 25
    }
    plt.rc('font', **font)

    figure_dimensions=(12, 8)

    plt.figure(figsize=figure_dimensions)


def main():
    if os.name == 'nt':
        root = 'E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\src\\app\\new_rnn\\out\\models\\'
    else:
        root = '/home/echen/Desktop/Projects/Sealer_NN/src/app/new_rnn/out/models/'

    terminal_char = dir_utils.get_terminal_directory_character()

    output_folder = root + ".." + terminal_char + "aggregate" + terminal_char + "training_aggregate" + terminal_char

    dir_utils.mkdir(output_folder)

    lstm_cell_directories = os.listdir(root)
    replicates = 10

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

    headers = np.array(["", "lstm_cells", "seed_range", "best_epoch", "valid_acc", "train_acc", "valid_loss", "train_loss"])
    numeric_headers = headers[3:]
    data_list = [headers]

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
                               chosen_train_accuracy, chosen_valid_loss, chosen_train_loss]

                        data_list.append(row)
                        row_id += 1

    data = np.array(data_list)
    df = pd.DataFrame(data=data[1:, 1:], index=data[1:,0], columns=data[0, 1:])

    df[numeric_headers] = df[numeric_headers].apply(pd.to_numeric)

    set_up_plot()
    sns.boxplot(data=df, x="seed_range", y="valid_acc", hue="lstm_cells")
    fig = plt.savefig(output_folder + "valid_acc.png")
    plt.close(fig)

    set_up_plot()
    sns.boxplot(data=df, x="seed_range", y="train_acc", hue="lstm_cells")
    fig = plt.savefig(output_folder + "train_acc.png")
    plt.close(fig)

    set_up_plot()
    sns.boxplot(data=df, x="seed_range", y="valid_loss", hue="lstm_cells")
    fig = plt.savefig(output_folder + "valid_loss.png")
    plt.close(fig)

    set_up_plot()
    sns.boxplot(data=df, x="seed_range", y="train_loss", hue="lstm_cells")
    fig = plt.savefig(output_folder + "train_loss.png")
    plt.close(fig)

    set_up_plot()
    sns.boxplot(data=df, x="seed_range", y="best_epoch", hue="lstm_cells")
    fig = plt.savefig(output_folder + "best_epoch.png")
    plt.close(fig)

if __name__ == "__main__":
    main()