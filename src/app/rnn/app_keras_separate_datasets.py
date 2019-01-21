import sys
sys.path.append('../../')

import time

import app.app_helper as helper
import app.rnn.app_helper as rnn_helper
from predict.rnn.KerasLSTMModel import KerasLSTMModel


def main():
    include_reverse_complement = True
    input_length = 26
    bases_to_predict = 1
    spacing = 0
    unique = False

    training_paths = ['../data/ecoli_contigs/ecoli_contig_1000.fastq']
    input_kmers_train, output_kmers_train = helper.extract_kmers(training_paths, input_length, spacing, bases_to_predict, include_reverse_complement, unique)
    input_seq_train, output_seq_train, shifted_output_train, train_input_stats_map = \
        helper.label_integer_encode_kmers(input_kmers_train, output_kmers_train, with_shifted_output=True)

    validation_paths = ['../data/ecoli_contigs/ecoli-400-600.fastq']
    input_kmers_valid, output_kmers_valid = helper.extract_kmers(validation_paths, input_length, spacing, bases_to_predict, include_reverse_complement, unique)
    input_seq_valid, output_seq_valid, shifted_output_valid, valid_input_stats_map = \
        helper.label_integer_encode_kmers(input_kmers_valid, output_kmers_valid, with_shifted_output=False)

    print("Encoding training set")
    input_one_hot_cube_train = helper.encode(input_length, input_seq_train)
    output_one_hot_cube_train = helper.encode(bases_to_predict, output_seq_train)
    shifted_output_seq_cube_train = helper.encode(input_length, shifted_output_train)
    print("Encoding validation set")
    input_one_hot_cube_valid = helper.encode(input_length, input_seq_valid)
    output_one_hot_cube_valid = helper.encode(bases_to_predict, output_seq_valid)

    model = KerasLSTMModel(prediction_length=bases_to_predict, batch_size=256, epochs=10, latent_dim=100)

    print("Computing input statistics...")
    print("Unique mappings: " + str(train_input_stats_map.get_total_unique_mappings_per_input()))

    start_time = time.time()
    model.fit(input_one_hot_cube_train, output_one_hot_cube_train, shifted_output_seq_cube_train)
    model.save_weights('../weights/my_model_weights.h5')
    end_time = time.time()
    print("Fitting took " + str(end_time - start_time) + "s")

    print()
    print("Output stats: " + str(train_input_stats_map.get_output_stats()))
    print()

    print("Predicting training set")
    rnn_helper.predict_and_validate(input_one_hot_cube_train, output_one_hot_cube_train, model, bases_to_predict)
    print("Predicting validation set")
    rnn_helper.predict_and_validate(input_one_hot_cube_valid, output_one_hot_cube_valid, model, bases_to_predict)


if __name__ == "__main__":
    main()

