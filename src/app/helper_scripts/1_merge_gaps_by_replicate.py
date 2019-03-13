import os

import utils.directory_utils as UTILS


def get_aggregate_fasta_name(replicate, rnn_dim):
    return "replicate_" + str(replicate) + "_" + "LD_" + str(rnn_dim) + "_gaps.fasta"

def main():
    if os.name == 'nt':
        root = 'E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\src\\app\\new_rnn\\out\\models\\'
    else:
        root = '/home/echen/Desktop/Projects/Sealer_NN/src/app/new_rnn/out/models/'

    terminal_char = UTILS.get_terminal_directory_character()

    output_folder = root + ".." + terminal_char + "aggregate" + terminal_char + "predicted_sequences" + terminal_char
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    rnn_dim_directories = os.listdir(root)
    replicates = 2

    ids = set()
    for rnn_dim in rnn_dim_directories:
        dim_path = root + rnn_dim + terminal_char
        experiments = os.listdir(dim_path)
        for id in experiments:
            fasta_id = id.split("_R_")[0]
            ids.add(fasta_id)

    for i in range(replicates):
        for rnn_dim in rnn_dim_directories:
            file = open(output_folder + get_aggregate_fasta_name(i, rnn_dim), "a")
            for id in ids:
                folder = id + "_R_" + str(i)
                folder_path = root + rnn_dim + terminal_char + folder + terminal_char
                results_files = os.listdir(folder_path)
                for results_file in results_files:
                    if results_file == "gap_predict_align.fa":
                        with open(folder_path + results_file) as fasta:
                            acc = 0
                            for line in fasta:
                                if acc < 4:
                                    file.write(line)
                                acc += 1
        file.close()

if __name__ == "__main__":
    main()
