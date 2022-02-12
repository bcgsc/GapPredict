import argparse
import os

arg_parser = argparse.ArgumentParser(description="GapPredict - LSTM Character Level Language Model for Gap Filling a Single Gap")
arg_parser.add_argument('-o', nargs=1, help="output directory", required=True)
arg_parser.add_argument('-fa', nargs=1, help="FASTA file for flanks and gaps, assumed that sequence 0 and 1 are flanks",
                        required=True)
arg_parser.add_argument('-fq', nargs=1, help="FASTQ file with reads mapping to flanks and gaps", required=True)
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
arg_parser.add_argument('-bl', type=int, nargs=1, default=[64], help="beam length")

args = arg_parser.parse_args()

gpu = str(args.gpu[0])

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

import utils.directory_utils as dir_utils
import train_model as train
import predict_greedy as greedy
import predict_beam_search as beam_search
from tensorflow.keras import backend as K
import tensorflow as tf

def reset_states():
    K.clear_session()
    tf.compat.v1.reset_default_graph()

def main(args):
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    base_output_directory = dir_utils.clean_directory_string(args.o[0])

    dir_utils.mkdir(base_output_directory)

    ref_file = args.fa[0]
    read_file = args.fq[0]

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
    beam_length = args.bl[0]

    terminal_directory_character = dir_utils.get_terminal_directory_character()
    gap_id = ref_file.split(terminal_directory_character)[-1].split(".")[0]

    base_output_directory += gap_id

    for i in range(replicates):
        replicate_num = i + replicate_offset
        output_directory = dir_utils.clean_directory_string(base_output_directory + "_R_" + str(replicate_num))

        dir_utils.mkdir(output_directory)

        print("Training model")
        train.train_model(output_directory, min_seed_length, ref_file, read_file, epochs, [batch_size], [rnn_dim], [embedding_dim], 1, patience, seed_range_upper)

        print("Greedy prediction")
        greedy.predict_reference(output_directory, gap_id, ref_file, embedding_dim, rnn_dim, min_seed_length, base_path=output_directory)
        greedy.predict_arbitrary_length(output_directory, gap_id, ref_file, embedding_dim, rnn_dim, prediction_length, base_path=output_directory)

        print("Beam search prediction")
        beam_search.predict_reference(output_directory, gap_id, ref_file, embedding_dim, rnn_dim, min_seed_length,
                          beam_length, base_path=output_directory)
        beam_search.predict_arbitrary_length(output_directory, gap_id, ref_file, embedding_dim, rnn_dim, prediction_length,
                                 beam_length, base_path=output_directory)

        reset_states()

if __name__ == "__main__":
    main(args)