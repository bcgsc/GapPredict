import sys
sys.path.append('../../')

import argparse
import os
import utils.directory_utils as UTILS

from app.new_rnn.train_model import train_model
from app.new_rnn.predict_by_reference import predict_reference
from app.new_rnn.predict_arbitrary_length import predict_arbitrary_length

def main():
    arg_parser = argparse.ArgumentParser(description="Help placeholder")
    arg_parser.add_argument('-o', nargs=1, help="output directory", required=True)
    arg_parser.add_argument('-fa', nargs=1, help="FASTA file for flanks and gaps, assumed that sequence 0 and 1 are flanks", required=True)
    arg_parser.add_argument('-fq', nargs=1, help="FASTQ file with reads mapping to flanks and gaps", required=True)
    #TODO: might change this to any value between 256 and 512
    arg_parser.add_argument('-hu', type=int, nargs=1, default=[512], help="number of hidden units in the LSTM")
    arg_parser.add_argument('-ed', type=int, nargs=1, default=[128], help="number of dimensions in base embedding vector")
    arg_parser.add_argument('-bs', type=int, nargs=1, default=[128], help="batch sizes")
    arg_parser.add_argument('-r', type=int, nargs=1, default=[1], help="# replicates of models to train")
    arg_parser.add_argument('-e', type=int, nargs=1, default=[1000], help="training epochs")
    arg_parser.add_argument('-sl', type=int, nargs=1, default=[26], help="minimum seed length for training")
    arg_parser.add_argument('-gpu', type=int, nargs=1, default=[0], help="GPU device ID to use")
    arg_parser.add_argument('-pl', type=int, nargs=1, default=[750], help="prediction length")

    args = arg_parser.parse_args()
    base_output_directory = UTILS.clean_directory_string(args.o[0])

    if not os.path.exists(base_output_directory):
        os.makedirs(base_output_directory)

    ref_file = args.fa[0]
    read_file = args.fq[0]

    batch_size = args.bs[0]
    rnn_dim = args.hu[0]
    embedding_dim = args.ed[0]
    epochs = args.e[0]
    min_seed_length = args.sl[0]
    replicates = args.r[0]
    prediction_length = args.pl[0]

    gpu = str(args.gpu[0])
    terminal_directory_character = UTILS.get_terminal_directory_character()
    id = ref_file.split(terminal_directory_character)[-1].split(".")[0]

    base_output_directory += id

    for i in range(replicates):
        output_directory = UTILS.clean_directory_string(base_output_directory + "_R_" + str(i))
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        train_model(output_directory, min_seed_length, ref_file, read_file, epochs, [batch_size], [rnn_dim], [embedding_dim], 1, gpu=gpu)
        predict_reference(output_directory, ref_file, embedding_dim, rnn_dim, min_seed_length, base_path=output_directory, gpu=gpu)
        predict_arbitrary_length(output_directory, id, ref_file, embedding_dim, rnn_dim, prediction_length, base_path=output_directory, gpu=gpu)

if __name__ == "__main__":
    main()