import os

import matplotlib.pyplot as plt
import numpy as np

import utils.directory_utils as dir_utils


def set_up_plot(is_training):
    epochs = 1000
    plt.rc('xtick', labelsize=30)
    plt.rc('ytick', labelsize=30)
    font = {
        'size': 40
    }
    plt.rc('font', **font)

    figure_dimensions=(18, 12)

    plt.figure(figsize=figure_dimensions)
    plt.ylim(0, 1.1)
    plt.xlim(0, epochs)
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy' if is_training else 'Validation Accuracy')

def main():
    if os.name == 'nt':
        root = 'E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\src\\app\\new_rnn\\out\\models\\'
    else:
        root = '/home/echen/Desktop/Projects/Sealer_NN/src/app/new_rnn/out/models/'

    terminal_char = dir_utils.get_terminal_directory_character()

    output_folder = root + ".." + terminal_char + "aggregate" + terminal_char + "training_curves" + terminal_char

    dir_utils.mkdir(output_folder)

    rnn_dim_directories = os.listdir(root)
    replicates = 2

    ids = set()
    for rnn_dim in rnn_dim_directories:
        dim_path = root + rnn_dim + terminal_char
        experiments = os.listdir(dim_path)
        for id in experiments:
            fasta_id = id.split("_R_")[0]
            ids.add(fasta_id)

    training_file = "training.npy"
    validation_file = "weighted_mean.npy"

    linewidth = 5
    alpha = 0.7
    colours = {"384_SL_52": 'r', "384_SL_104": 'orange', "512_SL_52": 'c', "512_SL_104": 'blue'}

    for id in ids:
        for i in range(replicates):
            set_up_plot(True)
            plot_id = id + "_training_R_" + str(i)
            for rnn_dim in rnn_dim_directories:
                folder = id + "_R_" + str(i)
                folder_path = root + rnn_dim + terminal_char + folder + terminal_char
                model_files = os.listdir(folder_path)
                for results_file in model_files:
                    if results_file.startswith("BS"):
                        training_folder = folder_path + results_file + terminal_char
                        training_metrics = np.load(training_folder + training_file)
                        training_epochs = np.arange(len(training_metrics))
                        plt.plot(training_epochs, training_metrics, linewidth=linewidth, alpha=alpha, color=colours[rnn_dim], label="LD " + str(rnn_dim))

                        validation_metrics = np.load(training_folder + validation_file)
                        best_epoch = np.argmax(validation_metrics)
                        plt.axvline(best_epoch, linewidth=linewidth, alpha=alpha, color=colours[rnn_dim])
            plt.legend(loc="best")
            fig = plt.savefig(output_folder + plot_id)
            plt.close(fig)

    for id in ids:
        for i in range(replicates):
            set_up_plot(False)
            plot_id = id + "_validation_R_" + str(i)
            for rnn_dim in rnn_dim_directories:
                folder = id + "_R_" + str(i)
                folder_path = root + rnn_dim + terminal_char + folder + terminal_char
                model_files = os.listdir(folder_path)
                for results_file in model_files:
                    if results_file.startswith("BS"):
                        training_folder = folder_path + results_file + terminal_char
                        validation_metrics = np.load(training_folder + validation_file)
                        validation_epochs = np.arange(len(validation_metrics))
                        plt.plot(validation_epochs, validation_metrics, linewidth=linewidth, alpha=alpha, color=colours[rnn_dim], label="LD " + str(rnn_dim))

                        best_epoch = np.argmax(validation_metrics)
                        plt.axvline(best_epoch, linewidth=linewidth, alpha=alpha, color=colours[rnn_dim])
            plt.legend(loc="best")
            fig = plt.savefig(output_folder + plot_id)
            plt.close(fig)

if __name__ == "__main__":
    main()
