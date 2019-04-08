import argparse
import os
import sys

arg_parser = argparse.ArgumentParser(description="Help placeholder")
arg_parser.add_argument('-f', nargs=1, help="input directory", required=True)
arg_parser.add_argument('-o', nargs=1, help="output directory", required=True)
arg_parser.add_argument('-hu', type=int, nargs=1, default=[512], help="number of hidden units in the LSTM")
arg_parser.add_argument('-ed', type=int, nargs=1, default=[128], help="number of dimensions in base embedding vector")
arg_parser.add_argument('-bs', type=int, nargs=1, default=[128], help="batch sizes")
arg_parser.add_argument('-r', type=int, nargs=1, default=[1], help="# replicates of models to train")
arg_parser.add_argument('-e', type=int, nargs=1, default=[1000], help="training epochs")
arg_parser.add_argument('-sl', type=int, nargs=1, default=[52], help="minimum seed length for training")
arg_parser.add_argument('-sr', type=int, nargs=1, default=[None], help="maximum seed extension for training, leave as default to go as long as the read")
arg_parser.add_argument('-gpu', type=int, nargs=1, default=[0], help="GPU device ID to use")
arg_parser.add_argument('-pl', type=int, nargs=1, default=[750], help="prediction length")
arg_parser.add_argument('-es', type=int, nargs=1, default=[200], help="early stopping patience epochs")
arg_parser.add_argument('-os', type=int, nargs=1, default=[0], help="replicate offset")
arg_parser.add_argument('-p', type=int, nargs=1, default=[1], help="partition to take")

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
from app.new_rnn.train_model import train_model
from app.new_rnn.predict_by_reference import predict_reference
from app.new_rnn.predict_arbitrary_length import predict_arbitrary_length
from keras import backend as K
import tensorflow as tf

def reset_states():
    K.clear_session()
    tf.reset_default_graph()

def main(args):
    base_directory = dir_utils.clean_directory_string(args.f[0])
    base_output_directory = dir_utils.clean_directory_string(args.o[0])
    batch_size = args.bs[0]
    rnn_dim = args.hu[0]
    embedding_dim = args.ed[0]
    epochs = args.e[0]
    min_seed_length = args.sl[0]
    replicates = args.r[0]
    prediction_length = args.pl[0]
    patience = args.es[0]
    seed_range_upper = args.sr[0]
    replicate_offset = args.os[0]
    partition = args.p[0]

    terminal_directory_character = dir_utils.get_terminal_directory_character()

    gaps = os.listdir(base_directory)
    gaps_with_size = []
    for gap in gaps:
        inner_directory = base_directory + gap + terminal_directory_character
        read_file = inner_directory + gap + ".fastq"
        statinfo = os.stat(read_file)
        gaps_with_size.append({
            'gap': gap,
            'size': statinfo.st_size
        })

    gaps_with_size.sort(key=lambda x: x['size'])

    partitions = []
    num_partitions = 3
    for i in range(num_partitions):
        partitions.append([])

    for i in range(len(gaps_with_size)):
        gap = gaps_with_size[i]['gap']
        partition_idx = i % num_partitions
        partitions[partition_idx].append(gap)
    gaps_to_train_on = partitions[partition-1]

    existing_models = os.listdir(base_output_directory)

    for gap in gaps_to_train_on:
        if gap in existing_models:
            continue
        inner_directory = base_directory + gap + terminal_directory_character
        output_directory = base_output_directory + gap + terminal_directory_character
        dir_utils.mkdir(output_directory)
        ref_file = inner_directory + gap + ".fasta"
        read_file = inner_directory + gap + ".fastq"
        gap_id = ref_file.split(terminal_directory_character)[-1].split(".")[0]

        for i in range(replicates):
            replicate_num = i + replicate_offset
            replicate_output_directory = dir_utils.clean_directory_string(output_directory + gap + "_R_" + str(replicate_num))

            dir_utils.mkdir(replicate_output_directory)

            train_model(replicate_output_directory, min_seed_length, ref_file, read_file, epochs, [batch_size], [rnn_dim],
                        [embedding_dim], 1, patience, seed_range_upper)
            predict_reference(replicate_output_directory, ref_file, embedding_dim, rnn_dim, min_seed_length, gap_id,
                              base_path=replicate_output_directory)
            predict_arbitrary_length(replicate_output_directory, gap_id, ref_file, embedding_dim, rnn_dim, prediction_length,
                                     base_path=replicate_output_directory)

            reset_states()



if __name__ == "__main__":
    main(args)