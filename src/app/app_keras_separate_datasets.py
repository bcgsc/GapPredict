import time

import app.app_helper as helper
from predict.KerasRNNModel import KerasRNNModel


def main():
    include_reverse_complement = True
    input_length = 26
    bases_to_predict = 1
    spacing = 0
    k = 1
    has_quality = False
    unique = False
    fill_in_the_blanks = False

    training_paths = ['data/ecoli_contigs/ecoli-0-400.fastq', 'data/ecoli_contigs/ecoli-600-1000.fastq']
    input_seq_train, input_quality_train, output_seq_train, shifted_output_train, train_input_stats_map = helper.extract_read_matrix(training_paths,
                                                                                                    input_length,
                                                                                                    spacing,
                                                                                                    bases_to_predict,
                                                                                                    include_reverse_complement,
                                                                                                    unique, fill_in_the_blanks)

    validation_paths = ['data/ecoli_contigs/ecoli-400-600.fastq']
    input_seq_valid, input_quality_valid, output_seq_valid, shifted_output_valid, valid_input_stats_map = helper.extract_read_matrix(validation_paths,
                                                                                                    input_length,
                                                                                                    spacing,
                                                                                                    bases_to_predict,
                                                                                                    include_reverse_complement,
                                                                                                    unique)
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

    model = KerasRNNModel(has_quality=has_quality, prediction_length=k, batch_size=64, epochs=10, latent_dim=100)

    start_time = time.clock()
    model.fit(input_one_hot_cube_train, output_one_hot_cube_train, shifted_output_seq_cube_train)
    end_time = time.clock()
    print("Fitting took " + str(end_time - start_time) + "s")

    print()
    print("Computing input statistics...")
    print("Redundant mappings: " + str(train_input_stats_map.get_inputs_with_redundant_mappings()))
    print("Output stats: " + str(train_input_stats_map.get_output_stats()))
    print()

    print("Predicting training set")
    helper.predict_and_validate(input_one_hot_cube_train, output_one_hot_cube_train, model, bases_to_predict)
    print("Predicting validation set")
    helper.predict_and_validate(input_one_hot_cube_valid, output_one_hot_cube_valid, model, bases_to_predict)


main()
