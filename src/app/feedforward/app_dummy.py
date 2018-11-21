import sys
sys.path.append('../../')

import time

import numpy as np

import app.rnn.app_helper as helper
from KmerLabelEncoder import KmerLabelEncoder
from SequenceImporter import SequenceImporter
from SequenceMatchCalculator import SequenceMatchCalculator
from SlidingWindowExtractor import SlidingWindowExtractor
from onehot.OneHotVector import OneHotVectorEncoder, OneHotVectorDecoder
from predict.feedforward.RandomPredictModel import RandomPredictModel


def extract_read_matrix(paths, input_length, spacing, bases_to_predict, include_reverse_complement):
    importer = SequenceImporter()
    extractor = SlidingWindowExtractor(input_length, spacing, bases_to_predict)
    encoder = KmerLabelEncoder()

    start_time = time.time()
    reads = importer.import_fastq(paths, include_reverse_complement)
    end_time = time.time()
    print("Import took " + str(end_time - start_time) + "s")

    start_time = time.time()
    input_kmers, output_kmers, quality_vectors = extractor.extract_kmers_from_sequence(reads)
    end_time = time.time()
    print("Extraction took " + str(end_time - start_time) + "s")

    start_time = time.time()
    helper.get_stats(input_kmers, output_kmers)
    end_time = time.time()
    print("Stats took " + str(end_time - start_time) + "s")

    start_time = time.time()
    input_seq, input_quality, output_seq, shifted_output_seq = encoder.encode_kmers(input_kmers, output_kmers, quality_vectors)
    end_time = time.time()
    print("Label Integer Encoding took " + str(end_time - start_time) + "s")
    return input_seq, input_quality, output_seq

def encode_reads(paths, input_length, spacing, bases_to_predict):
    extract_reverse_complement = True
    input_encoder = OneHotVectorEncoder(input_length)
    output_encoder = OneHotVectorEncoder(bases_to_predict)
    input_seq, input_quality, output_seq = extract_read_matrix(paths, input_length, spacing, bases_to_predict, extract_reverse_complement)

    start_time = time.time()
    input_one_hot_cube = input_encoder.encode_sequences(input_seq)
    end_time = time.time()
    print("Input one-hot encoding took " + str(end_time - start_time) + "s")

    start_time = time.time()
    output_one_hot_cube = output_encoder.encode_sequences(output_seq)
    end_time = time.time()
    print("Output one-hot encoding took " + str(end_time - start_time) + "s")
    return input_one_hot_cube, output_one_hot_cube

def main():
    global_start_time = time.time()
    input_length = 50
    bases_to_predict = 2
    spacing = 0

    match_calculator = SequenceMatchCalculator()

    arguments = sys.argv[1:]
    paths = arguments if len(arguments) > 0 else ['../data/read_1_1000.fastq', '../data/read_2_1000.fastq']
    input_one_hot_cube, output_one_hot_cube = encode_reads(paths, input_length, spacing, bases_to_predict)

    output_decoder = OneHotVectorDecoder(bases_to_predict)
    model = RandomPredictModel(bases_to_predict)

    start_time = time.time()
    model.fit(input_one_hot_cube, output_one_hot_cube)
    end_time = time.time()
    print("Fitting took " + str(end_time - start_time) + "s")

    start_time = time.time()
    predicted_output = model.predict(input_one_hot_cube)
    end_time = time.time()
    print("Predicting took " + str(end_time - start_time) + "s")

    start_time = time.time()
    decoded_predicted_output = output_decoder.decode_sequences(predicted_output)
    decoded_actual_output = output_decoder.decode_sequences(output_one_hot_cube)
    end_time = time.time()
    print("Decoding took " + str(end_time - start_time) + "s")

    start_time = time.time()
    matches = match_calculator.compare_sequences(decoded_predicted_output, decoded_actual_output, bases_to_check=bases_to_predict)

    mean_match = np.mean(matches, axis=0)
    print("Mean Match = " + str(mean_match))
    end_time = time.time()
    print("Validation took " + str(end_time - start_time) + "s")
    print("Total time = " + str(end_time - global_start_time) + "s")


if __name__ == "__main__":
    main()