import sys
sys.path.append('../')

import time

import numpy as np

import app.app_helper as helper
from KmerLabelEncoder import KmerLabelEncoder
from SequenceImporter import SequenceImporter
from SequenceMatchCalculator import SequenceMatchCalculator
from SlidingWindowExtractor import SlidingWindowExtractor
from onehot.OneHotMatrix import OneHotMatrixEncoder, OneHotMatrixDecoder
from predict.RandomPredictModel import RandomPredictModel


def extract_read_matrix(paths, input_length, spacing, bases_to_predict, include_reverse_complement):
    importer = SequenceImporter()
    extractor = SlidingWindowExtractor(input_length, spacing, bases_to_predict)
    encoder = KmerLabelEncoder()

    start_time = time.clock()
    reads = importer.import_fastq(paths, include_reverse_complement)
    end_time = time.clock()
    print("Import took " + str(end_time - start_time) + "s")

    start_time = time.clock()
    input_kmers, output_kmers, quality_vectors = extractor.extract_kmers_from_sequence(reads)
    end_time = time.clock()
    print("Extraction took " + str(end_time - start_time) + "s")

    start_time = time.clock()
    helper.get_stats(input_kmers, output_kmers)
    end_time = time.clock()
    print("Stats took " + str(end_time - start_time) + "s")

    start_time = time.clock()
    input_seq, input_quality, output_seq, shifted_output_seq = encoder.encode_kmers(input_kmers, output_kmers, quality_vectors, fill_in_the_blanks=True)
    end_time = time.clock()
    print("Label Integer Encoding took " + str(end_time - start_time) + "s")
    return input_seq, input_quality, output_seq

def encode_reads(paths, input_length, spacing, bases_to_predict):
    extract_reverse_complement = True
    input_encoder = OneHotMatrixEncoder(input_length, bases_to_predict)
    output_encoder = OneHotMatrixEncoder(input_length + bases_to_predict)
    input_seq, input_quality, output_seq = extract_read_matrix(paths, input_length, spacing, bases_to_predict, extract_reverse_complement)

    start_time = time.clock()
    input_one_hot_cube = input_encoder.encode_sequences(input_seq, input_quality)
    end_time = time.clock()
    print("Input one-hot encoding took " + str(end_time - start_time) + "s")

    start_time = time.clock()
    output_one_hot_cube = output_encoder.encode_sequences(output_seq)
    end_time = time.clock()
    print("Output one-hot encoding took " + str(end_time - start_time) + "s")
    return input_one_hot_cube, output_one_hot_cube

global_start_time = time.clock()
input_length = 50
bases_to_predict = 2
spacing = 0

match_calculator = SequenceMatchCalculator()

paths = ['data/read_1_1000.fastq', 'data/read_2_1000.fastq']
input_one_hot_cube, output_one_hot_cube = encode_reads(paths, input_length, spacing, bases_to_predict)

output_decoder = OneHotMatrixDecoder(input_length + bases_to_predict)
model = RandomPredictModel(bases_to_predict)

start_time = time.clock()
model.fit(input_one_hot_cube, output_one_hot_cube)
end_time = time.clock()
print("Fitting took " + str(end_time - start_time) + "s")

start_time = time.clock()
predicted_output = model.predict(input_one_hot_cube)
end_time = time.clock()
print("Predicting took " + str(end_time - start_time) + "s")

start_time = time.clock()
decoded_predicted_output = output_decoder.decode_sequences(predicted_output)
decoded_actual_output = output_decoder.decode_sequences(output_one_hot_cube)
end_time = time.clock()
print("Decoding took " + str(end_time - start_time) + "s")

start_time = time.clock()
total_bases = input_length + bases_to_predict
start_idx = total_bases - bases_to_predict
matches = match_calculator.compare_sequences(decoded_predicted_output, decoded_actual_output, start_idx=start_idx, bases_to_check=bases_to_predict)

mean_match = np.mean(matches, axis=0)
print("Mean Match = " + str(mean_match))
end_time = time.clock()
print("Validation took " + str(end_time - start_time) + "s")
print("Total time = " + str(end_time - global_start_time) + "s")