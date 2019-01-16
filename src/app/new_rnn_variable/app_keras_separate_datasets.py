import sys
sys.path.append('../../')

import time

import app.new_rnn_variable.app_helper as helper
from predict.new_rnn.BaseKerasLSTMModel import BaseKerasLSTMModel


def main():
    include_reverse_complement = True
    input_length = 26
    bases_to_predict = 1
    spacing = 0

    training_paths = ['../data/ecoli_contigs/ecoli_contig_1000.fastq']
    reads_train = helper.import_reads(training_paths, include_reverse_complement)
    input_kmers_train, output_kmers_train = helper.extract_kmers(reads_train, input_length, spacing)
    input_seq_train, output_seq_train, k_high_train, input_stats_map_train = \
        helper.label_integer_encode_kmers(input_kmers_train, output_kmers_train)

    validation_paths = ['../data/ecoli_contigs/ecoli-400-600.fastq']
    reads_valid = helper.import_reads(validation_paths, include_reverse_complement)
    input_kmers_valid, output_kmers_valid = helper.extract_kmers(reads_valid, input_length, spacing)
    input_seq_valid, output_seq_valid, k_high_valid, input_stats_map_valid = \
        helper.label_integer_encode_kmers(input_kmers_valid, output_kmers_valid)

    k_high = max(k_high_train, k_high_valid)
    print("k_high = " + str(k_high))

    print("Encoding training set")
    input_one_hot_cube_train = helper.encode(k_high, input_seq_train)
    output_one_hot_cube_train = helper.encode(bases_to_predict, output_seq_train, two_dim=True)
    print("Encoding validation set")
    input_one_hot_cube_valid = helper.encode(k_high, input_seq_valid)
    output_one_hot_cube_valid = helper.encode(bases_to_predict, output_seq_valid, two_dim=True)

    model = BaseKerasLSTMModel(batch_size=1024, epochs=5, latent_dim=100, with_gpu=True)

    print("Computing input statistics...")
    print("Unique mappings: " + str(input_stats_map_train.get_total_unique_mappings_per_input()))

    start_time = time.time()
    model.fit(input_one_hot_cube_train, output_one_hot_cube_train)
    model.save_weights('../weights/my_model_weights.h5')
    end_time = time.time()
    print("Fitting took " + str(end_time - start_time) + "s")

    print()
    print("Output stats: " + str(input_stats_map_train.get_output_stats()))
    print()

    print("Predicting training set")
    helper.predict_and_validate(input_one_hot_cube_train, output_one_hot_cube_train, model)
    print("Predicting validation set")
    helper.predict_and_validate(input_one_hot_cube_valid, output_one_hot_cube_valid, model)

if __name__ == "__main__":
    main()

