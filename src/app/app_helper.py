import time

import numpy as np

from KmerLabelEncoder import KmerLabelEncoder
from SequenceImporter import SequenceImporter
from SequenceMatchCalculator import SequenceMatchCalculator
from SlidingWindowExtractor import SlidingWindowExtractor
from onehot.OneHotMatrix import OneHotMatrixEncoder, OneHotMatrixDecoder
from stats.InputOutputFrequencyMap import InputOutputFrequencyMap


def get_stats(inputs, outputs):
    freq_map = InputOutputFrequencyMap()
    freq_map.load_input_outputs(inputs, outputs)
    print(str(freq_map.get_inputs_with_redundant_mappings()))
    return freq_map

def extract_read_matrix(paths, input_length, spacing, bases_to_predict, include_reverse_complement, unique, fill_in_the_blanks):
    importer = SequenceImporter()
    extractor = SlidingWindowExtractor(input_length, spacing, bases_to_predict)
    encoder = KmerLabelEncoder()

    start_time = time.clock()
    reads = importer.import_fastq(paths, include_reverse_complement)
    end_time = time.clock()
    print("Import took " + str(end_time - start_time) + "s")

    start_time = time.clock()
    input_kmers, output_kmers, quality_vectors = extractor.extract_kmers_from_sequence(reads, unique=unique)
    end_time = time.clock()
    print("Extraction took " + str(end_time - start_time) + "s")

    start_time = time.clock()
    input_stats_map = get_stats(input_kmers, output_kmers)
    end_time = time.clock()
    print("Stats took " + str(end_time - start_time) + "s")

    start_time = time.clock()
    input_seq, input_quality, output_seq, shifted_output_seq = \
        encoder.encode_kmers(input_kmers, output_kmers, quality_vectors, fill_in_the_blanks=fill_in_the_blanks)
    end_time = time.clock()
    print("Label Integer Encoding took " + str(end_time - start_time) + "s")
    return input_seq, input_quality, output_seq, shifted_output_seq, input_stats_map

def encode_reads(input_length, bases_to_predict, input_seq, input_quality, output_seq, shifted_output_seq, has_quality=False):
    input_encoder = OneHotMatrixEncoder(input_length)
    output_encoder = OneHotMatrixEncoder(bases_to_predict)

    start_time = time.clock()
    input_one_hot_cube = input_encoder.encode_sequences(input_seq, input_quality if has_quality else None)
    end_time = time.clock()
    print("Input one-hot encoding took " + str(end_time - start_time) + "s")

    start_time = time.clock()
    output_one_hot_cube = output_encoder.encode_sequences(output_seq)
    end_time = time.clock()
    print("Output one-hot encoding took " + str(end_time - start_time) + "s")

    start_time = time.clock()
    shifted_output_seq_cube = output_encoder.encode_sequences(shifted_output_seq)
    end_time = time.clock()
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

def predict_and_validate(input, output_seq_cube, model, bases_to_predict):
    decoder = OneHotMatrixDecoder(bases_to_predict)
    validator = SequenceMatchCalculator()
    stats = {}
    start_time = time.clock()
    num_predictions = len(input)
    decoded_actual_output = decoder.decode_sequences(output_seq_cube)

    decoded_predictions = []
    for i in range(num_predictions):
        predicted_output = model.predict(input[i:i+1])
        decoded_predicted_output = decoder.decode_sequences(predicted_output)
        decoded_sequence = ''.join(decoded_predicted_output[0])
        if decoded_sequence not in stats:
            stats[decoded_sequence] = 1
        else:
            stats[decoded_sequence] += 1

        decoded_predictions.append(decoded_predicted_output)

    matches = validator.compare_sequences(decoded_predictions, decoded_actual_output)

    mean_match = np.mean(matches, axis=0)
    print("Mean Match = " + str(mean_match))
    print("Stats = " + str(stats))

    end_time = time.clock()
    print("Predicting and Validation took " + str(end_time - start_time) + "s")