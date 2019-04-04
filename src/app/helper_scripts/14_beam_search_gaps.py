import argparse
import os
import sys

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-gpu', type=int, nargs=1, default=[0], help="GPU device ID to use")
args = arg_parser.parse_args()

gpu = str(args.gpu[0])

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

if os.name == 'nt':
    sys.path.append('E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\src\\')
else:
    sys.path.append('/home/echen/Desktop/Projects/Sealer_NN/src/')

import utils.directory_utils as dir_utils
from app.new_rnn.predict_beam_search import predict_reference, predict_arbitrary_length
from keras import backend as K
import tensorflow as tf

def reset_states():
    K.clear_session()
    tf.reset_default_graph()

def main():
    terminal_char = dir_utils.get_terminal_directory_character()
    fixed=False
    gap_type = "fixed" if fixed else "unfixed"

    root = '/projects/btl/scratch/echen/April_2_Results_Backup/scratch/' + gap_type + terminal_char

    gap_ids = os.listdir(root)
    num_replicates = 1

    ids = set()
    for gap_id in gap_ids:
        ids.add(gap_id)

    embedding_dim = 128
    latent_dim = 512
    min_seed_length = 52
    prune_length = 64
    length_to_predict=750

    base_fasta_path = "/projects/btl/scratch/echen/sealer_alignment_smaller/random_sample/out/" + gap_type + terminal_char

    for gap_id in ids:
        gap_folder = root + gap_id + terminal_char
        for i in range(num_replicates):
            model_folder = gap_id + "_R_" + str(i)
            folder_path = gap_folder + model_folder + terminal_char
            files = os.listdir(folder_path)
            if len(files) == 1:
                continue

            weights_path = folder_path

            fasta_path = base_fasta_path + gap_id + terminal_char + gap_id + ".fasta"
            predict_reference(weights_path, fasta_path, embedding_dim, latent_dim, min_seed_length, gap_id, prune_length, base_path=folder_path)
            predict_arbitrary_length(weights_path, gap_id, fasta_path, embedding_dim, latent_dim, length_to_predict,
                                     prune_length, base_path=folder_path)
            reset_states()

if __name__ == "__main__":
    main()