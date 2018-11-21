import sys
sys.path.append('../../')

import time

from sklearn import model_selection

import app.rnn.app_helper as helper
from predict.KerasLSTMModel import KerasLSTMModel


def main():
    include_reverse_complement = True
    input_length = 26
    bases_to_predict = 1
    spacing = 0
    has_quality = False
    unique = False
    fill_in_the_blanks = False

    arguments = sys.argv[1:]
    paths = arguments if len(arguments) > 0 else ['../data/read_1_300.fastq']
    input_seq, input_quality, output_seq, shifted_output_seq, input_stats_map = helper.extract_read_matrix(paths, input_length, spacing,
                                                                                   bases_to_predict, include_reverse_complement, unique, fill_in_the_blanks)

    input_seq_train, input_seq_valid, input_quality_train, input_quality_valid, output_seq_train, output_seq_valid, shifted_output_train, shifted_output_valid = model_selection.train_test_split(
        input_seq, input_quality, output_seq, shifted_output_seq, test_size=0.15, random_state=123)
    print("Encoding training set")
    input_one_hot_cube_train, output_one_hot_cube_train, shifted_output_seq_cube_train = helper.encode_reads(input_length,
                                                                                                      bases_to_predict,
                                                                                                      input_seq_train,
                                                                                                      input_quality_train,
                                                                                                      output_seq_train,
                                                                                                      shifted_output_train,
                                                                                                      has_quality=has_quality)
    print("Encoding validation set")
    input_one_hot_cube_valid, output_one_hot_cube_valid, shifted_output_seq_cube_valid = helper.encode_reads(input_length,
                                                                                                      bases_to_predict,
                                                                                                      input_seq_valid,
                                                                                                      input_quality_valid,
                                                                                                      output_seq_valid,
                                                                                                      shifted_output_valid,
                                                                                                      has_quality=has_quality)

    model = KerasLSTMModel(has_quality=has_quality, prediction_length=bases_to_predict, batch_size=64, epochs=5, latent_dim=100)

    print("Computing input statistics...")
    print("Unique mappings: " + str(input_stats_map.get_total_unique_mappings_per_input()))

    start_time = time.time()
    model.fit(input_one_hot_cube_train, output_one_hot_cube_train, shifted_output_seq_cube_train)
    model.save_weights('../weights/my_model_weights.h5')
    end_time = time.time()
    print("Fitting took " + str(end_time - start_time) + "s")

    print()
    print("Output stats: " + str(input_stats_map.get_output_stats()))
    print()

    print("Predicting training set")
    helper.predict_and_validate(input_one_hot_cube_train, output_one_hot_cube_train, model, bases_to_predict)
    print("Predicting validation set")
    helper.predict_and_validate(input_one_hot_cube_valid, output_one_hot_cube_valid, model, bases_to_predict)


if __name__ == "__main__":
    main()
