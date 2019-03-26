import os
import sys

if os.name == 'nt':
    sys.path.append('E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\src\\')
else:
    sys.path.append('/home/echen/Desktop/Projects/Sealer_NN/src/')

import utils.directory_utils as dir_utils
from app.new_rnn.predict_by_reference import predict_reference

def main():
    if os.name == 'nt':
        root = 'E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\src\\app\\new_rnn\\out\\models\\'
    else:
        root = '/home/echen/Desktop/Projects/Sealer_NN/src/app/new_rnn/out/models/'

    terminal_char = dir_utils.get_terminal_directory_character()

    output_folder = root + ".." + terminal_char + "aggregate" + terminal_char

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

    if os.name == 'nt':
        fasta_path = 'E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\src\\app\\data\\real_gaps\\sealer_filled\\7391826_358-1408.fasta'
    else:
        fasta_path = "/home/echen/Desktop/Projects/Sealer_NN/src/app/data/real_gaps/sealer_filled/7391826_358-1408.fasta"

    embedding_dim = 128

    for id in ids:
        for lstm_cells in lstm_cell_directories:
            for i in range(replicates):
                folder = id + "_R_" + str(i)
                folder_path = root + lstm_cells + terminal_char + folder + terminal_char
                weights_path = folder_path

                tokens = lstm_cells.split("_")
                latent_dim = int(tokens[0])
                min_seed_length = int(tokens[2])

                predict_reference(weights_path, fasta_path, embedding_dim, latent_dim, min_seed_length, id, base_path=folder_path)

if __name__ == "__main__":
    main()