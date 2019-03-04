import sys

sys.path.append('../../')

import time
import os

import app.app_helper as helper
import numpy as np
import utils.directory_utils as UTILS
from predict.heuristic.HashModel import HashModel
from preprocess.SequenceMatchCalculator import SequenceMatchCalculator
from preprocess.SequenceImporter import SequenceImporter
from preprocess.SequenceReverser import SequenceReverser
from viz.SequenceRegenerationViz import SequenceRegenerationViz

def main():
    include_reverse_complement = True
    seed_length = 50
    bases_to_predict = 1
    length_to_predict = 750
    spacing = 0
    unique = False

    training_paths = ['../data/real_gaps/sealer_filled/7465348_13506-14596.fastq']
    input_kmers_train, output_kmers_train = helper.extract_kmers(training_paths, seed_length,
                                                                                        spacing, bases_to_predict,
                                                                                        include_reverse_complement,
                                                                                        unique)

    validator = SequenceMatchCalculator()
    reverser = SequenceReverser()
    model = HashModel(bases_to_predict)

    start_time = time.time()
    model.fit(input_kmers_train, output_kmers_train)
    end_time = time.time()
    print("Fitting took " + str(end_time - start_time) + "s")

    start_time = time.time()
    yhat_train = model.predict(input_kmers_train)
    end_time = time.time()
    print("Predicting took " + str(end_time - start_time) + "s")

    start_time = time.time()
    matches_train = validator.compare_sequences(yhat_train, output_kmers_train)
    mean_match_train = np.mean(matches_train, axis=0)
    print("Mean Match Training = " + str(mean_match_train))

    print("Training Check took " + str(end_time - start_time) + "s")

    # FASTA prediction
    importer = SequenceImporter()
    fasta_path = '../data/real_gaps/sealer_filled/7465348_13506-14596.fasta'
    terminal_directory_character = UTILS.get_terminal_directory_character()
    id = fasta_path.split(terminal_directory_character)[-1].split(".")[0]

    sequences = importer.import_fasta([fasta_path])
    left_flank = sequences[0]
    left_flank_rc = reverser.reverse_complement(left_flank)
    right_flank = sequences[1]
    right_flank_rc = reverser.reverse_complement(right_flank)

    predict(left_flank, model, seed_length)
    predict(left_flank_rc, model, seed_length)
    predict(right_flank, model, seed_length)
    predict(right_flank_rc, model, seed_length)

    forward = predict_arbitrary(left_flank, model, seed_length, length_to_predict)
    reverse_complement = predict_arbitrary(right_flank_rc, model, seed_length, length_to_predict)

    if os.name == 'nt':
        root_path = 'E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\src\\app\\heuristic\\out\\'
    else:
        root_path = '/home/echen/Desktop/Projects/Sealer_NN/src/app/heuristic/out/'
    viz = SequenceRegenerationViz(root_directory=root_path, directory=id)
    viz.save_complements(forward, reverse_complement, fasta_ref=fasta_path)

def predict(sequence, model, seed_length):
    validator = SequenceMatchCalculator()
    sequence_length = len(sequence)

    input = sequence[0:seed_length]
    bases_to_predict = sequence[seed_length:sequence_length]
    output_length = sequence_length - seed_length
    remaining_length = output_length

    current_sequence = str(input)
    lower_bound = 0
    upper_bound = lower_bound + seed_length
    while remaining_length > 0:
        seed = current_sequence[lower_bound:upper_bound]

        prediction = model.predict([seed])
        current_sequence += prediction[0]

        remaining_length -= 1
        lower_bound += 1
        upper_bound += 1

    predicted_sequence = current_sequence[seed_length:sequence_length]
    matches = validator.compare_sequences(predicted_sequence, bases_to_predict)
    mean_match = np.mean(matches)
    print(mean_match)

def predict_arbitrary(flank, model, seed_length, length_to_predict):
    remaining_length = length_to_predict

    current_sequence = str(flank)
    lower_bound = len(flank)-seed_length
    upper_bound = lower_bound + seed_length
    while remaining_length > 0:
        seed = current_sequence[lower_bound:upper_bound]

        prediction = model.predict([seed])
        current_sequence += prediction[0]

        remaining_length -= 1
        lower_bound += 1
        upper_bound += 1

    return current_sequence

if __name__ == "__main__":
    main()