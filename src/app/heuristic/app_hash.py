import sys

sys.path.append('../../')

import time

import app.app_helper as helper
from predict.heuristic.HashModel import HashModel

def main():
    include_reverse_complement = True
    input_length = 50
    bases_to_predict = 2
    spacing = 0
    has_quality = False
    unique = False
    as_matrix = False

    arguments = sys.argv[1:]
    paths = arguments if len(arguments) > 0 else ['../data/read_1_1000.fastq', '../data/read_2_1000.fastq']

if __name__ == "__main__":
    main()