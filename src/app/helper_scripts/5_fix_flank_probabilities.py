import os
import utils.directory_utils as UTILS

from app.new_rnn.predict_by_reference import predict_reference

def main():
    if os.name == 'nt':
        root = 'E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\src\\app\\new_rnn\\out\\models\\'
    else:
        root = '/home/echen/Desktop/Projects/Sealer_NN/src/app/new_rnn/out/models/'

    if os.name == 'nt':
        fasta_root = 'E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\src\\app\\data\\real_gaps\\sealer_filled\\'
    else:
        fasta_root = '/home/echen/Desktop/Projects/Sealer_NN/src/app/data/real_gaps/sealer_filled/'

    terminal_char = UTILS.get_terminal_directory_character()

    rnn_dim_directories = os.listdir(root)

    replicates = 3

    ids = set()
    for rnn_dim in rnn_dim_directories:
        dim_path = root + rnn_dim + terminal_char
        experiments = os.listdir(dim_path)
        for id in experiments:
            fasta_id = id.split("_R_")[0]
            ids.add(fasta_id)

    for i in range(replicates):
        for rnn_dim in rnn_dim_directories:
            for id in ids:
                folder = id + "_R_" + str(i)
                folder_path = root + rnn_dim + terminal_char + folder + terminal_char

                weights_path = folder_path
                fasta_path = fasta_root + id + ".fasta"
                embedding_dim = 128
                latent_dim = int(rnn_dim)
                min_seed_length = 26
                base_path = folder_path
                predict_reference(weights_path, fasta_path, embedding_dim, latent_dim, min_seed_length, id, base_path=base_path)

if __name__ == "__main__":
    main()
