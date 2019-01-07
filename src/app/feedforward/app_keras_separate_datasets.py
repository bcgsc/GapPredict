import sys

sys.path.append('../../')

import time

import app.app_helper as helper
from predict.feedforward.KerasVanillaModel import KerasVanillaModel
import numpy as np


def main():
    include_reverse_complement = True
    input_length = 26
    bases_to_predict = 1
    spacing = 0
    has_quality = False
    unique = False
    as_matrix = False

    #TODO: parametrize this
    training_paths = ['../data/ecoli_contigs/ecoli_contig_1000.fastq']
    input_kmers_train, output_kmers_train, quality_train = helper.extract_kmers(training_paths, input_length, spacing, bases_to_predict, include_reverse_complement, unique)
    input_seq_train, input_quality_train, output_seq_train, shifted_output_train, train_input_stats_map = \
        helper.label_integer_encode_kmers(input_kmers_train, output_kmers_train, quality_train, with_shifted_output=False)

    validation_paths = ['../data/ecoli_contigs/ecoli-400-600.fastq']
    input_kmers_valid, output_kmers_valid, quality_valid = helper.extract_kmers(validation_paths, input_length, spacing, bases_to_predict, include_reverse_complement, unique)
    input_seq_valid, input_quality_valid, output_seq_valid, shifted_output_valid, valid_input_stats_map = \
        helper.label_integer_encode_kmers(input_kmers_valid, output_kmers_valid, quality_valid, with_shifted_output=False)

    print("Encoding training set")
    input_one_hot_cube_train = helper.encode(input_length, input_seq_train, input_quality_train, has_quality=has_quality, as_matrix=as_matrix)
    output_one_hot_cube_train = helper.encode(bases_to_predict, output_seq_train, None, has_quality=has_quality, as_matrix=as_matrix)
    print("Encoding validation set")
    input_one_hot_cube_valid = helper.encode(input_length, input_seq_valid, input_quality_valid, has_quality=has_quality, as_matrix=as_matrix)
    output_one_hot_cube_valid = helper.encode(bases_to_predict, output_seq_valid, None, has_quality=has_quality, as_matrix=as_matrix)

    model = KerasVanillaModel(input_length, bases_to_predict, batch_size=512, epochs=10)

    print("Computing input statistics...")
    print("Unique mappings: " + str(train_input_stats_map.get_total_unique_mappings_per_input()))

    start_time = time.time()
    model.fit(input_one_hot_cube_train, output_one_hot_cube_train)
    model.save_weights('../weights/my_model_weights.h5')
    end_time = time.time()
    print("Fitting took " + str(end_time - start_time) + "s")

    print()
    print("Output stats: " + str(train_input_stats_map.get_output_stats()))
    print()

    print("Predicting training set")
    helper.predict_and_validate(input_one_hot_cube_train, output_one_hot_cube_train, model, bases_to_predict,
                                as_matrix=as_matrix)
    print("Predicting validation set")
    helper.predict_and_validate(input_one_hot_cube_valid, output_one_hot_cube_valid, model, bases_to_predict,
                                as_matrix=as_matrix)


if __name__ == "__main__":
    main()
