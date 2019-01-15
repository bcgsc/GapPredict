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
    paths = arguments if len(arguments) > 0 else ['../data/read_1_300.fastq']
    reads = helper.import_reads(paths, include_reverse_complement)

    reads_train, reads_valid = model_selection.train_test_split(reads, test_size=0.15, random_state=123)

    print("Training set")
    input_kmers_train, output_kmers_train, k_high_train = helper.extract_kmers(reads_train, input_length, spacing)
    input_seq_train, output_seq_train, input_stats_map_train = \
        helper.label_integer_encode_kmers(input_kmers_train, output_kmers_train)
    print("Validation set")
    input_kmers_valid, output_kmers_valid, k_high_valid = helper.extract_kmers(reads_valid, input_length, spacing)
    input_seq_valid, output_seq_valid, input_stats_map_valid = \
        helper.label_integer_encode_kmers(input_kmers_valid, output_kmers_valid)

    k_high = max(k_high_train, k_high_valid)

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
