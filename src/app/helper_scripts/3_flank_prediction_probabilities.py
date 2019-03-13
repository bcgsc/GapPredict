import os

import utils.directory_utils as UTILS
import matplotlib.pyplot as plt
import numpy as np
from viz.SequenceRegenerationViz import SequenceRegenerationViz

def set_up_plot():
    max_length = 1400
    plt.rc('xtick', labelsize=30)
    plt.rc('ytick', labelsize=30)
    font = {
        'size': 40
    }
    plt.rc('font', **font)

    figure_dimensions=(18, 12)

    plt.figure(figsize=figure_dimensions)
    plt.ylim(0, 1.1)
    plt.xlim(0, max_length)
    plt.xlabel('Position')
    plt.ylabel('Top Base Probability')

def read_error_positions(file, reverse=False):
    txt = open(file, 'r')
    for i in range(3):
        txt.readline()
    matches_line = txt.readline()
    if reverse:
        matches_line = matches_line[::-1]
    mismatches = []
    for i in range(len(matches_line)):
        if matches_line[i] == "X":
            mismatches.append(i)
    txt.close()
    return np.array(mismatches)

def main():
    viz = SequenceRegenerationViz()
    if os.name == 'nt':
        root = 'E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\src\\app\\new_rnn\\out\\models\\'
    else:
        root = '/home/echen/Desktop/Projects/Sealer_NN/src/app/new_rnn/out/models/'

    terminal_char = UTILS.get_terminal_directory_character()

    output_folder = root + ".." + terminal_char + "aggregate" + terminal_char + "flank_prediction" + terminal_char
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

    file_name = "predicted_probabilities.npy"
    txt_file_name = "align.txt"

    linewidth = 5
    alpha = 0.7
    error_alpha = 0.2
    lf = "left_flank"
    rf = "right_flank"
    f = "forward"
    rc = "reverse_complement"

    gap_lengths = { #TODO: hardcoded = bad
        "7465325_7588-8680": 130,
        "7465348_13506-14596": 93,
        "7473236_14236-15319": 104,
        "7473263_4934-6078": 353,
        "7505257_43822-44849": 102
    }

    font = {
        'size': 35
    }

    colours = {"384_SL_52": 'r', "384_SL_104": 'orange', "512_SL_52": 'c', "512_SL_104": 'blue'}

    for id in ids:
        for i in range(replicates):
            set_up_plot()
            plot_id = id + "_flanks" + "_R_" + str(i) + "_F"
            for rnn_dim in rnn_dim_directories:
                seed_length = int(rnn_dim.split("_")[2])
                folder = id + "_R_" + str(i)
                folder_path = root + rnn_dim + terminal_char + folder + terminal_char + "regenerate_seq" + terminal_char
                left_flank_forward = np.max(np.load(folder_path + lf + terminal_char + f + terminal_char + file_name), axis=1)
                right_flank_forward = np.max(np.load(folder_path + rf + terminal_char + f + terminal_char + file_name), axis=1)

                avg_lff = viz.sliding_window_average(left_flank_forward)
                avg_rff = viz.sliding_window_average(right_flank_forward)

                left_flank_length = len(left_flank_forward)
                right_flank_length = len(right_flank_forward)

                left_flank_pos = np.arange(left_flank_length) + seed_length

                gap_length = gap_lengths[id]
                right_flank_start = seed_length + left_flank_length + gap_length
                right_flank_pos = np.arange(right_flank_length) + right_flank_start + seed_length

                left_flank_forward_mismatches = read_error_positions(folder_path + lf + terminal_char + f + terminal_char + txt_file_name)
                right_flank_forward_mismatches = read_error_positions(folder_path + rf + terminal_char + f + terminal_char + txt_file_name) + right_flank_start
                all_mismatches = np.concatenate([left_flank_forward_mismatches, right_flank_forward_mismatches])

                plt.plot(left_flank_pos, avg_lff, linewidth=linewidth, alpha=alpha, color=colours[rnn_dim], label="LD_" + str(rnn_dim)+"_F")
                plt.plot(right_flank_pos, avg_rff, linewidth=linewidth, alpha=alpha, color=colours[rnn_dim])

                for mismatch in all_mismatches:
                    plt.axvline(mismatch, linewidth=linewidth, alpha=error_alpha, color=colours[rnn_dim])
            plt.legend(loc="best", prop=font)
            fig = plt.savefig(output_folder + plot_id)
            plt.close(fig)

    for id in ids:
        for i in range(replicates):
            set_up_plot()
            plot_id = id + "_flanks" + "_R_" + str(i) + "_RC"
            for rnn_dim in rnn_dim_directories:
                seed_length = int(rnn_dim.split("_")[2])
                folder = id + "_R_" + str(i)
                folder_path = root + rnn_dim + terminal_char + folder + terminal_char + "regenerate_seq" + terminal_char
                left_flank_rc = np.max(np.load(folder_path + lf + terminal_char + rc + terminal_char + file_name), axis=1)[::-1]
                right_flank_rc = np.max(np.load(folder_path + rf + terminal_char + rc + terminal_char + file_name), axis=1)[::-1]

                avg_lfrc = viz.sliding_window_average(left_flank_rc)
                avg_rfrc = viz.sliding_window_average(right_flank_rc)

                left_flank_length = len(left_flank_rc)
                right_flank_length = len(right_flank_rc)

                left_flank_pos = np.arange(left_flank_length)

                gap_length = gap_lengths[id]
                right_flank_start = seed_length + left_flank_length + gap_length
                right_flank_pos = np.arange(right_flank_length) + right_flank_start

                left_flank_forward_mismatches = read_error_positions(folder_path + lf + terminal_char + rc + terminal_char + txt_file_name, reverse=True)
                right_flank_forward_mismatches = read_error_positions(folder_path + rf + terminal_char + rc + terminal_char + txt_file_name, reverse=True) + right_flank_start
                all_mismatches = np.concatenate([left_flank_forward_mismatches, right_flank_forward_mismatches])

                plt.plot(left_flank_pos, avg_lfrc, linewidth=linewidth, alpha=alpha, color=colours[rnn_dim], label="LD_" + str(rnn_dim)+"_RC")
                plt.plot(right_flank_pos, avg_rfrc, linewidth=linewidth, alpha=alpha, color=colours[rnn_dim])

                for mismatch in all_mismatches:
                    plt.axvline(mismatch, linewidth=linewidth, alpha=error_alpha, color=colours[rnn_dim])
            plt.legend(loc="best", prop=font)
            fig = plt.savefig(output_folder + plot_id)
            plt.close(fig)

if __name__ == "__main__":
    main()
