import sys
sys.path.append('../../')

import time

from sklearn import model_selection

import app.new_rnn.app_helper as helper
from predict.new_rnn.BaseKerasLSTMModel import BaseKerasLSTMModel


def main():
    include_reverse_complement = True
    input_length = 26
    bases_to_predict = 1
    spacing = 0

    arguments = sys.argv[1:]
    paths = arguments if len(arguments) > 0 else ['../data/ecoli_contigs/ecoli_contig_1000.fastq']
    reads = helper.import_reads(paths, include_reverse_complement)

    reads_train, reads_valid = model_selection.train_test_split(reads, test_size=0.10, random_state=123)

    input_kmers_valid, output_kmers_valid, k_high_valid = helper.extract_kmers(reads_valid, input_length, spacing)
    input_seq_valid, output_seq_valid, input_stats_map_valid = \
        helper.label_integer_encode_kmers(input_kmers_valid, output_kmers_valid)
    #hack to get a random shuffle of some size TODO: make this use numpy random so we can do 0% or 100% as well

    print("Encoding data set")
    input_one_hot_cube_valid = helper.encode(input_length, input_seq_valid)
    output_one_hot_cube_valid = helper.encode(bases_to_predict, output_seq_valid, two_dim=True)

    model = BaseKerasLSTMModel(batch_size=1024, epochs=5, latent_dim=100, with_gpu=True)

    start_time = time.time()
    model.load_weights('../weights/my_model_weights.h5')
    end_time = time.time()
    print("Loading weights took " + str(end_time - start_time) + "s")

    print()
    print("Computing input statistics...")
    print("Unique mappings: " + str(input_stats_map_valid.get_total_unique_mappings_per_input()))
    print()
    print("Output stats: " + str(input_stats_map_valid.get_output_stats()))
    print()

    print("Predicting data set")
    helper.predict_and_validate(input_one_hot_cube_valid, output_one_hot_cube_valid, model)


if __name__ == "__main__":
    main()

