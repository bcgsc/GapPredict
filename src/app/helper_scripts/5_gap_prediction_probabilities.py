import os

import utils.directory_utils as UTILS
import matplotlib.pyplot as plt
import numpy as np
from viz.SequenceRegenerationViz import SequenceRegenerationViz

def set_up_plot():
    max_length = 1500
    plt.rc('xtick', labelsize=28)
    plt.rc('ytick', labelsize=28)
    font = {
        'size': 30
    }
    plt.rc('font', **font)

    figure_dimensions=(18, 12)

    plt.figure(figsize=figure_dimensions)
    plt.ylim(0, 1.1)
    plt.xlim(-500, max_length)
    plt.xlabel('Position')
    plt.ylabel('Top Base Probability')

def main():
    viz = SequenceRegenerationViz()
    if os.name == 'nt':
        root = 'E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\src\\app\\new_rnn\\out\\models\\'
    else:
        root = '/home/echen/Desktop/Projects/Sealer_NN/src/app/new_rnn/out/models/'

    terminal_char = UTILS.get_terminal_directory_character()

    output_folder = root + ".." + terminal_char + "aggregate" + terminal_char + "gap_prediction" + terminal_char
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

    file_name = "predicted_probabilities.npy"

    linewidth = 5
    alpha = 0.7
    f = "forward"
    rc = "reverse_complement"

    gap_lengths = { #TODO: hardcoded = bad
        "7465325_7588-8680": 130,
        "7465348_13506-14596": 93,
        "7473236_14236-15319": 104,
        "7473263_4934-6078": 353,
        "7505257_43822-44849": 102
    }
    left_flank_length = 500 #TODO:hardcoded

    font = {
        'size': 18
    }

    for id in ids:
        for i in range(replicates):
            set_up_plot()
            plot_id = id + "_gaps" + "_R_" + str(i)
            for rnn_dim in rnn_dim_directories:
                folder = id + "_R_" + str(i)
                folder_path = root + rnn_dim + terminal_char + folder + terminal_char + "predict_gap" + terminal_char
                forward = np.max(np.load(folder_path + f + terminal_char + file_name), axis=1)
                reverse_complement = np.max(np.load(folder_path + rc + terminal_char + file_name), axis=1)[::-1]

                avg_f = viz.sliding_window_average(forward)
                avg_rc = viz.sliding_window_average(reverse_complement)

                forward_length = len(forward)
                rc_length = len(reverse_complement)

                forward_pos = np.arange(forward_length) + left_flank_length

                gap_length = gap_lengths[id]
                rc_start = left_flank_length + gap_length
                rc_pos = np.arange(rc_start - rc_length, rc_start)

                plt.plot(forward_pos, avg_f, linewidth=linewidth, alpha=alpha, label="LD_" + str(rnn_dim)+"_F")
                plt.plot(rc_pos, avg_rc, linewidth=linewidth, alpha=alpha, label="LD_" + str(rnn_dim)+"_RC")
            plt.legend(loc="best", prop=font)
            fig = plt.savefig(output_folder + plot_id)
            plt.close(fig)
    #TODO: maybe add vertical lines for the first mismatch each model makes

if __name__ == "__main__":
    main()
