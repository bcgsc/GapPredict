import time

import numpy as np

from KmerLabelEncoder import KmerLabelEncoder
from SequenceImporter import SequenceImporter
from SequenceMatchCalculator import SequenceMatchCalculator
from SlidingWindowExtractor import SlidingWindowExtractor
from onehot.OneHotMatrix import OneHotMatrixEncoder, OneHotMatrixDecoder
from predict.RandomPredictModel import RandomPredictModel
from stats.InputOutputFrequencyMap import InputOutputFrequencyMap


def get_stats(inputs, outputs):
    freq_map = InputOutputFrequencyMap()
    freq_map.load_input_outputs(inputs, outputs)
    print(str(freq_map.get_inputs_with_redundant_mappings()))

def extract_read_matrix(paths, input_length, spacing, bases_to_predict):
    importer = SequenceImporter()
    extractor = SlidingWindowExtractor(input_length, spacing, bases_to_predict)
    encoder = KmerLabelEncoder()

    start_time = time.clock()
    reads = importer.import_fastq(paths, True)
    end_time = time.clock()
    print("Import took " + str(end_time - start_time) + "s")

    start_time = time.clock()
    input_kmers, output_kmers, quality_vectors = extractor.extract_kmers_from_sequence(reads)
    end_time = time.clock()
    print("Extraction took " + str(end_time - start_time) + "s")

    start_time = time.clock()
    get_stats(input_kmers, output_kmers)
    end_time = time.clock()
    print("Stats took " + str(end_time - start_time) + "s")

    start_time = time.clock()
    input_seq, input_quality, output_seq, shifted_output_seq = encoder.encode_kmers(input_kmers, output_kmers, quality_vectors, fill_in_the_blanks=True)
    end_time = time.clock()
    print("Label Encoding took " + str(end_time - start_time) + "s")
    return input_seq, input_quality, output_seq

def encode_reads(paths, input_length, spacing, bases_to_predict):
    input_encoder = OneHotMatrixEncoder(input_length, bases_to_predict)
    output_encoder = OneHotMatrixEncoder(input_length + bases_to_predict)
    input_seq, input_quality, output_seq = extract_read_matrix(paths, input_length, spacing, bases_to_predict)

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
bases_to_predict = 1
spacing = 0
k = 1

match_calculator = SequenceMatchCalculator()

paths = ['data/read_1_1000.fastq', 'data/read_2_1000.fastq']
input_one_hot_cube, output_one_hot_cube = encode_reads(paths, input_length, spacing, bases_to_predict)

output_decoder = OneHotMatrixDecoder(input_length + bases_to_predict)
model = RandomPredictModel(k)

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
num_predictions = len(decoded_predicted_output)
match_score = np.zeros(num_predictions)

for i in range(num_predictions):
    predicted_sequence = decoded_predicted_output[i]
    actual_sequence = decoded_actual_output[i]
    assert len(predicted_sequence) == len(actual_sequence)

    total_bases = len(actual_sequence)
    bases_to_check = 1
    start_idx = total_bases - bases_to_check
    num_mismatches = match_calculator.compare_sequences(predicted_sequence, actual_sequence, start_idx=start_idx, bases_to_check=bases_to_check)
    match_score[i] = bases_to_check - num_mismatches

mean_match = np.mean(match_score)
print("Mean Match = " + str(mean_match))
end_time = time.clock()
print("Validation took " + str(end_time - start_time) + "s")
print("Total time = " + str(end_time - global_start_time) + "s")