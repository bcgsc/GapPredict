import time

import numpy as np

import app.app_helper as base_helper
import constants.VariableLengthRnnEncodingConstants as CONSTANTS
from onehot.OneHotVector import OneHotVectorDecoder
from preprocess.SequenceImporter import SequenceImporter
from preprocess.SequenceMatchCalculator import SequenceMatchCalculator
from preprocess.VariableLengthKmerExtractor import VariableLengthKmerExtractor
from preprocess.VariableLengthKmerLabelEncoder import VariableLengthKmerLabelEncoder

#TODO: not the nicest thing to have
bases_to_predict = 1

def get_stats(inputs, outputs, verbose=False):
    return base_helper.get_stats(inputs, outputs, verbose=verbose)

def encode(length, seq, two_dim=False):
    encoded_reads = base_helper.encode(length, seq, encoding_constants=CONSTANTS)
    if two_dim:
        # TODO: maybe not the most elegant way of doing it
        return encoded_reads.squeeze()
    else:
        return encoded_reads

def validate_kmer(kmer):
    return base_helper.validate_kmer(kmer, bases_to_predict)

def import_reads(paths, include_reverse_complement):
    importer = SequenceImporter()

    start_time = time.time()
    reads = importer.import_fastq(paths, include_reverse_complement)
    end_time = time.time()
    print("Import took " + str(end_time - start_time) + "s")

    return reads

def extract_kmers(reads, k_low, spacing):
    extractor = VariableLengthKmerExtractor(k_low, spacing, bases_to_predict)

    start_time = time.time()
    input_kmers, output_kmers = extractor.extract_kmers_from_sequence(reads)
    end_time = time.time()
    print("Extraction took " + str(end_time - start_time) + "s")
    return input_kmers, output_kmers

def label_integer_encode_kmers(input_kmers, output_kmers, verbose=False):
    encoder = VariableLengthKmerLabelEncoder()

    start_time = time.time()
    input_stats_map = get_stats(input_kmers, output_kmers, verbose)
    end_time = time.time()
    print("Stats took " + str(end_time - start_time) + "s")

    start_time = time.time()
    input_seq, output_seq, k_high = encoder.encode_kmers(input_kmers, output_kmers)
    end_time = time.time()
    print("Label Integer Encoding took " + str(end_time - start_time) + "s")
    return input_seq, output_seq, k_high, input_stats_map

def predict_and_validate(input, output_seq_cube, model):
    decoder = OneHotVectorDecoder(bases_to_predict, encoding_constants=CONSTANTS)
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
    if bases_to_predict > 1:
        overall_mean_match = np.mean(matches)
        print("Overall Mean Match = " + str(overall_mean_match))

    end_time = time.time()

    print("Validation took " + str(end_time - start_time) + "s")