import time

import numpy as np
from sklearn import model_selection

from SequenceImporter import SequenceImporter
from SequenceMatchCalculator import SequenceMatchCalculator
from SlidingWindowExtractor import SlidingWindowExtractor
from onehot.OneHotMatrix import OneHotMatrixEncoder, OneHotMatrixDecoder
from predict.KerasRNNModel import KerasRNNModel


def extract_read_matrix(paths, input_length, spacing, bases_to_predict):
    importer = SequenceImporter()
    extractor = SlidingWindowExtractor(input_length, spacing, bases_to_predict)

    start_time = time.clock()
    reads = importer.import_fastq(paths, True)
    end_time = time.clock()
    print("Import took " + str(end_time - start_time) + "s")

    start_time = time.clock()
    input_seq, input_quality, output_seq, shifted_output_seq = extractor.extract_input_output_from_sequence(reads, False)
    end_time = time.clock()
    print("Extraction took " + str(end_time - start_time) + "s")
    return input_seq, input_quality, output_seq, shifted_output_seq


def encode_reads(input_length, bases_to_predict, input_seq, input_quality, output_seq, shifted_output_seq):
    input_encoder = OneHotMatrixEncoder(input_length)
    output_encoder = OneHotMatrixEncoder(bases_to_predict)
    start_time = time.clock()
    input_one_hot_cube = input_encoder.encode_sequences(input_seq, input_quality)
    end_time = time.clock()
    print("Input encoding took " + str(end_time - start_time) + "s")

    start_time = time.clock()
    output_one_hot_cube = output_encoder.encode_sequences(output_seq)
    end_time = time.clock()
    print("Output encoding took " + str(end_time - start_time) + "s")

    start_time = time.clock()
    shifted_output_seq_cube = output_encoder.encode_sequences(shifted_output_seq)
    end_time = time.clock()
    print("Shifted output encoding took " + str(end_time - start_time) + "s")
    return input_one_hot_cube, output_one_hot_cube, output_seq, shifted_output_seq_cube


def validate_sequences(predicted_sequence, actual_sequence, validator):
    assert len(predicted_sequence) == len(actual_sequence)

    total_bases = len(actual_sequence)
    bases_to_check = 1
    start_idx = total_bases - bases_to_check
    num_mismatches = validator.compare_sequences(predicted_sequence, actual_sequence,
                                                 start_idx=start_idx, bases_to_check=bases_to_check)
    return bases_to_check - num_mismatches


def predict_and_validate(input, output_seq, model, decoder, validator):
    start_time = time.clock()
    num_predictions = len(input)
    match_score = np.zeros(num_predictions)

    #TODO: I need to rethink how to implement predict - right now things are definitely not working because predict isn't like what a DNN would do
    for i in range(num_predictions):
        predicted_output = model.predict(input[i:i+1])
        decoded_predicted_output = decoder.decode_sequences(predicted_output)
        actual_sequence = output_seq[i]
        match_score[i] = validate_sequences(decoded_predicted_output, actual_sequence, validator)

    mean_match = np.mean(match_score)
    print("Mean Match = " + str(mean_match))

    end_time = time.clock()
    print("Predicting and Validation took " + str(end_time - start_time) + "s")


def main():
    input_length = 50
    bases_to_predict = 1
    spacing = 0
    k = 1

    match_calculator = SequenceMatchCalculator()

    paths = ['data/read_1_1000.fastq', 'data/read_2_1000.fastq']
    input_seq, input_quality, output_seq, shifted_output_seq = extract_read_matrix(paths, input_length, spacing,
                                                                                   bases_to_predict)
    #TODO: kind of long...
    input_seq_train, input_seq_valid, input_quality_train, input_quality_valid, output_seq_train, output_seq_valid, shifted_output_train, shifted_output_valid = model_selection.train_test_split(input_seq, input_quality, output_seq, shifted_output_seq, test_size=0.15, random_state=123)
    print("Encoding training set")
    input_one_hot_cube_train, output_one_hot_cube_train, output_seq_train, shifted_output_seq_cube_train = encode_reads(input_length, bases_to_predict, input_seq_train, input_quality_train, output_seq_train, shifted_output_train)
    print("Encoding validation set")
    input_one_hot_cube_valid, output_one_hot_cube_valid, output_seq_valid, shifted_output_seq_cube_valid = encode_reads(input_length, bases_to_predict, input_seq_valid, input_quality_valid, output_seq_valid, shifted_output_valid)

    output_decoder = OneHotMatrixDecoder(bases_to_predict)
    model = KerasRNNModel(has_quality=True, prediction_length=k, batch_size=64, epochs=10, latent_dim=100)

    start_time = time.clock()
    model.fit(input_one_hot_cube_train, output_one_hot_cube_train, shifted_output_seq_cube_train)
    end_time = time.clock()
    print("Fitting took " + str(end_time - start_time) + "s")

    print("Predicting training set")
    predict_and_validate(input_one_hot_cube_train, output_seq_train, model, output_decoder, match_calculator)
    print("Predicting validation set")
    predict_and_validate(input_one_hot_cube_valid, output_seq_valid, model, output_decoder, match_calculator)


main()