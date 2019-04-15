import argparse
import os
import sys
import time

arg_parser = argparse.ArgumentParser(description="Help placeholder")
arg_parser.add_argument('-f', nargs=1, help="input directory", required=True)
arg_parser.add_argument('-ref', nargs=1, help="reference directory", required=True)
arg_parser.add_argument('-hu', type=int, nargs=1, default=[512], help="number of hidden units in the LSTM")
arg_parser.add_argument('-ed', type=int, nargs=1, default=[128], help="number of dimensions in base embedding vector")
arg_parser.add_argument('-r', type=int, nargs=1, default=[1], help="# replicates of models to train")
arg_parser.add_argument('-sl', type=int, nargs=1, default=[52], help="minimum seed length for training")
arg_parser.add_argument('-gpu', type=int, nargs=1, default=[0], help="GPU device ID to use")
arg_parser.add_argument('-pl', type=int, nargs=1, default=[750], help="prediction length")
arg_parser.add_argument('-p', type=int, nargs=1, default=[1], help="partition to take")
arg_parser.add_argument('-pr', type=int, nargs=1, default=[64], help="prune length")

args = arg_parser.parse_args()

gpu = str(args.gpu[0])

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
#change this if someone is competing for GPU
os.environ["CUDA_VISIBLE_DEVICES"] = gpu;

if os.name == 'nt':
    sys.path.append('E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\src\\')
else:
    sys.path.append('/home/echen/Desktop/Projects/Sealer_NN/src/')

import utils.directory_utils as dir_utils
from app.new_rnn.predict_beam_search import predict_arbitrary_length, predict_reference
from keras import backend as K
import tensorflow as tf

def reset_states():
    K.clear_session()
    tf.reset_default_graph()

def main(args):
    base_directory = dir_utils.clean_directory_string(args.f[0])
    base_reference_directory = dir_utils.clean_directory_string(args.ref[0])
    latent_dim = args.hu[0]
    embedding_dim = args.ed[0]
    min_seed_length = args.sl[0]
    replicates = args.r[0]
    prediction_length = args.pl[0]
    partition = args.p[0]
    prune_length = args.pr[0]

    terminal_directory_character = dir_utils.get_terminal_directory_character()

    gaps = os.listdir(base_directory)

    gaps.sort()

    partitions = []
    num_partitions = 3
    for i in range(num_partitions):
        partitions.append([])

    for i in range(len(gaps)):
        gap = gaps[i]
        partition_idx = i % num_partitions
        partitions[partition_idx].append(gap)
    gaps_to_train_on = partitions[partition-1]

    for gap_id in gaps_to_train_on:
        inner_directory = base_directory + gap_id + terminal_directory_character

        for i in range(replicates):
            replicate_num = i
            replicate_output_directory = dir_utils.clean_directory_string(inner_directory + gap_id + "_R_" + str(replicate_num))
            model_files = os.listdir(replicate_output_directory)
            #if "beam_search" in model_files:
            #    continue

            fasta_path = base_reference_directory + gap_id + terminal_directory_character + gap_id + ".fasta"
            weights_path = replicate_output_directory

            start_time = time.time()
            predict_reference(weights_path, fasta_path, embedding_dim, latent_dim, min_seed_length, gap_id,
                              prune_length, base_path=replicate_output_directory)
            end_time = time.time()
            print("Beam search flank took " + str(end_time - start_time) + "s")
            start_time = time.time()
            predict_arbitrary_length(weights_path, gap_id, fasta_path, embedding_dim, latent_dim, prediction_length,
                                     prune_length, base_path=replicate_output_directory)
            end_time = time.time()
            print("Beam search gap took " + str(end_time - start_time) + "s")

            reset_states()



if __name__ == "__main__":
    main(args)