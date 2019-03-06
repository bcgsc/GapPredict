import os

import utils.directory_utils as UTILS
from preprocess.SequenceReverser import SequenceReverser


def get_aggregate_fasta_name(replicate, rnn_dim):
    return "replicate_" + str(replicate) + "_" + "LD_" + str(rnn_dim) + "_flanks.fasta"

def main():
    reverser = SequenceReverser()
    if os.name == 'nt':
        root = 'E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\src\\app\\new_rnn\\out\\models\\'
    else:
        root = '/home/echen/Desktop/Projects/Sealer_NN/src/app/new_rnn/out/models/'

    terminal_char = UTILS.get_terminal_directory_character()

    output_folder = root + ".." + terminal_char + "aggregate" + terminal_char + "predicted_sequences" + terminal_char
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

    for i in range(replicates):
        for rnn_dim in rnn_dim_directories:
            file = open(output_folder + get_aggregate_fasta_name(i, rnn_dim), "a")
            for id in ids:
                folder = id + "_R_" + str(i)
                folder_path = root + rnn_dim + terminal_char + folder + terminal_char + "regenerate_seq" + terminal_char
                flanks = os.listdir(folder_path)
                for flank in flanks:
                    flank_folder_path = folder_path + flank + terminal_char
                    if not os.path.isdir(flank_folder_path):
                        continue
                    flank_folders = os.listdir(flank_folder_path)
                    for strand in flank_folders:
                        strand_folder_path = flank_folder_path + strand + terminal_char
                        results_files = os.listdir(strand_folder_path)
                        for results_file in results_files:
                            if results_file == "align.txt":
                                with open(strand_folder_path + results_file) as align_txt:
                                    acc = 0
                                    for line in align_txt:
                                        if acc == 4:
                                            file.write(">" + id + "_" + flank + "_" + strand + "_LD_" + rnn_dim + "\n")
                                            if strand == "forward":
                                                file.write(line)
                                            elif strand == "reverse_complement":
                                                sequence = reverser.reverse_complement(line[0:len(line)-1])
                                                file.write(sequence + "\n")
                                        acc += 1
        file.close()




if __name__ == "__main__":
    main()
