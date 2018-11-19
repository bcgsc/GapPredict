import time

import numpy as np

from KmerLabelEncoder import KmerLabelEncoder
from SequenceImporter import SequenceImporter
from SequenceMatchCalculator import SequenceMatchCalculator
from SlidingWindowExtractor import SlidingWindowExtractor
from onehot.OneHotMatrix import OneHotMatrixEncoder, OneHotMatrixDecoder
from stats.InputOutputFrequencyMap import InputOutputFrequencyMap


def get_stats(inputs, outputs, verbose = False):
    freq_map = InputOutputFrequencyMap()
    freq_map.load_input_outputs(inputs, outputs)
    if verbose:
        print(str(freq_map.get_inputs_with_redundant_mappings()))
    return freq_map

def extract_read_matrix(paths, input_length, spacing, bases_to_predict, include_reverse_complement, unique, fill_in_the_blanks, verbose=False):
    importer = SequenceImporter()
    extractor = SlidingWindowExtractor(input_length, spacing, bases_to_predict)
    encoder = KmerLabelEncoder()

    start_time = time.time()
    reads = importer.import_fastq(paths, include_reverse_complement)
    end_time = time.time()
    print("Import took " + str(end_time - start_time) + "s")

    start_time = time.time()
    input_kmers, output_kmers, quality_vectors = extractor.extract_kmers_from_sequence(reads, unique=unique)
    end_time = time.time()
    print("Extraction took " + str(end_time - start_time) + "s")

    start_time = time.time()
    input_stats_map = get_stats(input_kmers, output_kmers, verbose)
    end_time = time.time()
    print("Stats took " + str(end_time - start_time) + "s")

    start_time = time.time()
    input_seq, input_quality, output_seq, shifted_output_seq = \
        encoder.encode_kmers(input_kmers, output_kmers, quality_vectors, fill_in_the_blanks=fill_in_the_blanks)
    end_time = time.time()
    print("Label Integer Encoding took " + str(end_time - start_time) + "s")
    return input_seq, input_quality, output_seq, shifted_output_seq, input_stats_map

def encode_reads(input_length, bases_to_predict, input_seq, input_quality, output_seq, shifted_output_seq, has_quality=False):
    input_encoder = OneHotMatrixEncoder(input_length)
    output_encoder = OneHotMatrixEncoder(bases_to_predict)

    start_time = time.time()
    input_one_hot_cube = input_encoder.encode_sequences(input_seq, input_quality if has_quality else None)
    end_time = time.time()
    print("Input one-hot encoding took " + str(end_time - start_time) + "s")

    start_time = time.time()
    output_one_hot_cube = output_encoder.encode_sequences(output_seq)
    end_time = time.time()
    print("Output one-hot encoding took " + str(end_time - start_time) + "s")

    start_time = time.time()
    shifted_output_seq_cube = output_encoder.encode_sequences(shifted_output_seq)
    end_time = time.time()
    print("Shifted one-hot output encoding took " + str(end_time - start_time) + "s")
    return input_one_hot_cube, output_one_hot_cube, shifted_output_seq_cube

def validate_sequences(predicted_sequence, actual_sequence, validator):
    assert len(predicted_sequence) == len(actual_sequence)

    total_bases = len(actual_sequence)
    bases_to_check = 1
    start_idx = total_bases - bases_to_check
    num_mismatches = validator.compare_sequences(predicted_sequence, actual_sequence,
                                                 start_idx=start_idx, bases_to_check=bases_to_check)
    return bases_to_check - num_mismatches

def get_checkpoints(num_predictions):
    progress_checks = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])*num_predictions
    progress_checks = progress_checks.astype(int)
    return set(progress_checks)

def predict(input, model, bases_to_predict):
    stats = {}
    decoded_predictions = []

    num_predictions = len(input)
    decoder = OneHotMatrixDecoder(bases_to_predict)
    progress_checks = get_checkpoints(num_predictions)

    for i in range(num_predictions):
        if i in progress_checks:
            print(str(int((i*100)/num_predictions)) + "% finished")
        predicted_output = model.predict(input[i:i + 1])
        decoded_predicted_output = decoder.decode_sequences(predicted_output)
        decoded_sequence = ''.join(decoded_predicted_output[0])
        if decoded_sequence not in stats:
            stats[decoded_sequence] = 1
        else:
            stats[decoded_sequence] += 1

        decoded_predictions.append(decoded_predicted_output[0])

    print("Stats = " + str(stats))
    return decoded_predictions

def predict_and_validate(input, output_seq_cube, model, bases_to_predict):
    decoder = OneHotMatrixDecoder(bases_to_predict)
    validator = SequenceMatchCalculator()

    start_time = time.time()
    decoded_actual_output = decoder.decode_sequences(output_seq_cube)

    decoded_predictions = predict(input, model, bases_to_predict)
    end_time = time.time()
    print("Predicting took " + str(end_time - start_time) + "s")

    start_time = time.time()
    matches = validator.compare_sequences(decoded_predictions, decoded_actual_output)

    mean_match = np.mean(matches, axis=0)
    print("Mean Match = " + str(mean_match))

    end_time = time.time()

    print("Validation took " + str(end_time - start_time) + "s")