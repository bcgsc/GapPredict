import os

import numpy as np

import utils.directory_utils as UTILS


def main():
    if os.name == 'nt':
        root = 'E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\src\\app\\new_rnn\\out\\models\\'
    else:
        root = '/home/echen/Desktop/Projects/Sealer_NN/src/app/new_rnn/out/models/'

    terminal_char = UTILS.get_terminal_directory_character()

    output_folder = root + ".." + terminal_char + "aggregate" + terminal_char + "training_curves" + terminal_char
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    rnn_dim_directories = os.listdir(root)
    replicates = 3

    ids = set()
    for rnn_dim in rnn_dim_directories:
        dim_path = root + rnn_dim + terminal_char
        experiments = os.listdir(dim_path)
        for id in experiments:
            fasta_id = id.split("_R_")[0]
            ids.add(fasta_id)

    validation_file = "validation.npy"

    for id in ids:
        for i in range(replicates):
            for rnn_dim in rnn_dim_directories:
                folder = id + "_R_" + str(i)
                folder_path = root + rnn_dim + terminal_char + folder + terminal_char
                model_files = os.listdir(folder_path)
                for results_file in model_files:
                    if results_file.startswith("BS"):
                        training_folder = folder_path + results_file + terminal_char
                        validation_metrics = np.load(training_folder + validation_file)
                        mean = np.mean(validation_metrics, axis=1)
                        np.save(training_folder + "weighted_mean", mean)
                        np.save(training_folder + "lengths", np.array([500, 500, 500, 500]))

if __name__ == "__main__":
    main()
