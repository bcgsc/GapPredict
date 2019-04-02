import argparse
import os
import sys

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-gpu', type=int, nargs=1, default=[0], help="GPU device ID to use")
arg_parser.add_argument('-p', type=int, nargs=1, default=[1], help="Partition (1-3)")
args = arg_parser.parse_args()

gpu = str(args.gpu[0])
partition = str(args.p[0])

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

if os.name == 'nt':
    sys.path.append('E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\src\\')
else:
    sys.path.append('/home/echen/Desktop/Projects/Sealer_NN/src/')

import utils.directory_utils as dir_utils
from app.new_rnn.predict_beam_search import predict_reference
from keras import backend as K
import tensorflow as tf

def reset_states():
    K.clear_session()
    tf.reset_default_graph()

def main():
    terminal_char = dir_utils.get_terminal_directory_character()

    root = '/projects/btl/scratch/echen/March_27_Results_Backup/scratch/' + partition + terminal_char

    lstm_cell_directories = os.listdir(root)
    num_replicates = 30

    ids = set()
    for lstm_cells in lstm_cell_directories:
        dim_path = root + lstm_cells + terminal_char
        replicates = os.listdir(dim_path)
        for gap_id in replicates:
            fasta_id = gap_id.split("_R_")[0]
            ids.add(fasta_id)

    fasta_path = "/home/echen/Desktop/Projects/Sealer_NN/src/app/data/real_gaps/sealer_filled/7391826_358-1408.fasta"

    embedding_dim = 128
    latent_dim = 512
    min_seed_length = 52
    prune_length = 64

    for gap_id in ids:
        for lstm_cells in lstm_cell_directories:
            for i in range(num_replicates):
                folder = gap_id + "_R_" + str(i)
                folder_path = root + lstm_cells + terminal_char + folder + terminal_char
                weights_path = folder_path

                predict_reference(weights_path, fasta_path, embedding_dim, latent_dim, min_seed_length, gap_id, prune_length, base_path=folder_path)
                reset_states()

if __name__ == "__main__":
    main()