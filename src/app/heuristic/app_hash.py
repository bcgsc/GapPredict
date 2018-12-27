import sys

sys.path.append('../../')

import time

import app.app_helper as helper
import numpy as np
from predict.heuristic.HashModel import HashModel
from preprocess.SequenceMatchCalculator import SequenceMatchCalculator

def main():
    include_reverse_complement = True
    input_length = 50
    bases_to_predict = 1
    spacing = 0
    unique = False

    training_paths = ['../data/ecoli_contigs/ecoli-0-400.fastq', '../data/ecoli_contigs/ecoli-600-1000.fastq']
    input_kmers_train, output_kmers_train, quality_vectors_train = helper.extract_kmers(training_paths, input_length,
                                                                                        spacing, bases_to_predict,
                                                                                        include_reverse_complement,
                                                                                        unique)
    validation_paths = ['../data/ecoli_contigs/ecoli-400-600.fastq']
    input_kmers_valid, output_kmers_valid, quality_vectors_valid = helper.extract_kmers(validation_paths, input_length, spacing,
                                                                      bases_to_predict, include_reverse_complement,
                                                                      unique)

    validator = SequenceMatchCalculator()
    model = HashModel(bases_to_predict)

    start_time = time.time()
    model.fit(input_kmers_train, output_kmers_train)
    end_time = time.time()
    print("Fitting took " + str(end_time - start_time) + "s")

    start_time = time.time()
    yhat_valid = model.predict(input_kmers_valid)
    yhat_train = model.predict(input_kmers_train)
    end_time = time.time()
    print("Predicting took " + str(end_time - start_time) + "s")

    start_time = time.time()
    matches_train = validator.compare_sequences(yhat_train, output_kmers_train)
    mean_match_train = np.mean(matches_train, axis=0)
    print("Mean Match Training = " + str(mean_match_train))

    matches_valid = validator.compare_sequences(yhat_valid, output_kmers_valid)
    mean_match_valid = np.mean(matches_valid, axis=0)
    print("Mean Match Validation = " + str(mean_match_valid))
    end_time = time.time()

    print("Validation took " + str(end_time - start_time) + "s")

if __name__ == "__main__":
    main()