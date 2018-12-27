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

    arguments = sys.argv[1:]
    paths = arguments if len(arguments) > 0 else ['../data/ecoli_contigs/ecoli_contig_1000.fastq']
    input_kmers, output_kmers, quality_vectors = helper.extract_kmers(paths, input_length, spacing, bases_to_predict, include_reverse_complement, unique)

    validator = SequenceMatchCalculator()
    model = HashModel()

    start_time = time.time()
    model.fit(input_kmers, output_kmers)
    end_time = time.time()
    print("Fitting took " + str(end_time - start_time) + "s")

    start_time = time.time()
    yhat = model.predict(input_kmers)
    end_time = time.time()
    print("Predicting took " + str(end_time - start_time) + "s")

    start_time = time.time()
    matches = validator.compare_sequences(yhat, output_kmers)
    mean_match = np.mean(matches, axis=0)
    print("Mean Match = " + str(mean_match))

    end_time = time.time()

    print("Validation took " + str(end_time - start_time) + "s")

if __name__ == "__main__":
    main()