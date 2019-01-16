import sys
sys.path.append('../../')

import time

from sklearn import model_selection

import app.app_helper as helper
import app.rnn.app_helper as rnn_helper
from predict.rnn.KerasLSTMModel import KerasLSTMModel


def main():
    include_reverse_complement = True
    input_length = 26
    bases_to_predict = 1
    spacing = 0
    unique = False

    arguments = sys.argv[1:]
    paths = arguments if len(arguments) > 0 else ['../data/read_1_300.fastq']
    input_kmers, output_kmers = helper.extract_kmers(paths, input_length, spacing, bases_to_predict, include_reverse_complement, unique)

    input_kmers_train, input_kmers_valid, output_kmers_train, output_kmers_valid = model_selection.train_test_split(
        input_kmers, output_kmers, test_size=0.15, random_state=123)

    print("Training set")
    input_seq_train, output_seq_train, shifted_output_seq_train, input_stats_map_train = \
        helper.label_integer_encode_kmers(input_kmers_train, output_kmers_train, with_shifted_output=True)
    print("Validation set")
    input_seq_valid, output_seq_valid, shifted_output_seq_valid, input_stats_map_valid = \
        helper.label_integer_encode_kmers(input_kmers_valid, output_kmers_valid, with_shifted_output=False)
    print("Encoding training set")
    input_one_hot_cube_train = helper.encode(input_length, input_seq_train)
    output_one_hot_cube_train = helper.encode(bases_to_predict, output_seq_train)
    shifted_output_seq_cube_train = helper.encode(input_length, shifted_output_seq_train)
    print("Encoding validation set")
    input_one_hot_cube_valid = helper.encode(input_length, input_seq_valid)
    output_one_hot_cube_valid = helper.encode(bases_to_predict, output_seq_valid)

    model = KerasLSTMModel(prediction_length=bases_to_predict, batch_size=64, epochs=5, latent_dim=100, with_gpu=True)

    print("Computing input statistics...")
    print("Unique mappings: " + str(input_stats_map_train.get_total_unique_mappings_per_input()))

    start_time = time.time()
    model.fit(input_one_hot_cube_train, output_one_hot_cube_train, shifted_output_seq_cube_train)
    model.save_weights('../weights/my_model_weights.h5')
    end_time = time.time()
    print("Fitting took " + str(end_time - start_time) + "s")

    print()
    print("Output stats: " + str(input_stats_map_train.get_output_stats()))
    print()

    print("Predicting training set")
    rnn_helper.predict_and_validate(input_one_hot_cube_train, output_one_hot_cube_train, model, bases_to_predict)
    print("Predicting validation set")
    rnn_helper.predict_and_validate(input_one_hot_cube_valid, output_one_hot_cube_valid, model, bases_to_predict)


if __name__ == "__main__":
    main()
