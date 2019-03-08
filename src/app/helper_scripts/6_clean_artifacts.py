import os
import utils.directory_utils as UTILS

def main():
    if os.name == 'nt':
        root = 'E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\src\\app\\new_rnn\\out\\models\\'
    else:
        root = '/home/echen/Desktop/Projects/Sealer_NN/src/app/new_rnn/out/models/'

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
                base_folder_path = root + rnn_dim + terminal_char + folder + terminal_char + "regenerate_seq" + terminal_char
                txt_file_path = base_folder_path + "bidirectional_align.txt"
                os.remove(txt_file_path)

                flank_folders = ["left_flank", "right_flank"]
                flank_type_folders = ["forward", "reverse_complement"]
                for flank in flank_folders:
                    flank_folder = base_folder_path + flank + terminal_char
                    for flank_type in flank_type_folders:
                        plot_file = flank_folder + flank_type + terminal_char + "sliding_window_probability.png"
                        os.remove(plot_file)


if __name__ == "__main__":
    main()