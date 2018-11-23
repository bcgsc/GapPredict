import time

import numpy as np

import constants.EncodingConstants as CONSTANTS
from KmerLabelEncoder import KmerLabelEncoder
from SequenceImporter import SequenceImporter
from SequenceMatchCalculator import SequenceMatchCalculator
from SlidingWindowExtractor import SlidingWindowExtractor
from onehot.OneHotMatrix import OneHotMatrixEncoder, OneHotMatrixDecoder
from onehot.OneHotVector import OneHotVectorEncoder, OneHotVectorDecoder
from stats.InputOutputFrequencyMap import InputOutputFrequencyMap


def get_stats(inputs, outputs, verbose=False):
    freq_map = InputOutputFrequencyMap()
    freq_map.load_input_outputs(inputs, outputs)
    if verbose:
        print(str(freq_map.get_inputs_with_redundant_mappings()))
    return freq_map


def extract_read_matrix(paths, input_length, spacing, bases_to_predict, include_reverse_complement, unique,
                        verbose=False, with_shifted_output=True):
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
        encoder.encode_kmers(input_kmers, output_kmers, quality_vectors, with_shifted_output)
    end_time = time.time()
    print("Label Integer Encoding took " + str(end_time - start_time) + "s")
    return input_seq, input_quality, output_seq, shifted_output_seq, input_stats_map

def encode(length, quality, has_quality=False, as_matrix=True):
    #TODO: implement this later so we aren't constrained to always pass in shifted_seq
    pass

def encode_reads(input_length, bases_to_predict, input_seq, input_quality, output_seq, shifted_output_seq,
                 has_quality=False, as_matrix=True):
    input_encoder = OneHotMatrixEncoder(input_length) if as_matrix else OneHotVectorEncoder(input_length)
    output_encoder = OneHotMatrixEncoder(bases_to_predict) if as_matrix else OneHotVectorEncoder(bases_to_predict)

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


def predict_and_validate(input, output_seq_cube, model, bases_to_predict, as_matrix=True):
    decoder = OneHotMatrixDecoder(bases_to_predict) if as_matrix else OneHotVectorDecoder(bases_to_predict)
    validator = SequenceMatchCalculator()

    start_time = time.time()

    predicted_output = model.predict(input)
    end_time = time.time()
    print("Predicting took " + str(end_time - start_time) + "s")

    start_time = time.time()

    decoded_predicted_output = decoder.decode_sequences(predicted_output)
    decoded_actual_output = decoder.decode_sequences(output_seq_cube)

    matches = validator.compare_sequences(decoded_predicted_output, decoded_actual_output)

    mean_match = np.mean(matches, axis=0)
    print("Mean Match = " + str(mean_match))

    end_time = time.time()

    print("Validation took " + str(end_time - start_time) + "s")

def validate_kmer(kmer, bases_to_predict):
    kmer_length = len(kmer)

    if kmer_length != bases_to_predict:
        return "Invalid kmer length"

    for l in range(kmer_length):
        if kmer[l] not in CONSTANTS.BASES:
            return "Invalid kmer base"

    return None