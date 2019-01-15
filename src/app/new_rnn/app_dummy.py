import sys
sys.path.append('../../')

import time

import app.new_rnn.app_helper as helper
from predict.new_rnn.RandomPredictModel import RandomPredictModel

def main():
    include_reverse_complement = True
    input_length = 50
    bases_to_predict = 1
    spacing = 0

    arguments = sys.argv[1:]
    paths = arguments if len(arguments) > 0 else ['../data/read_1_300.fastq']

    reads = helper.import_reads(paths, include_reverse_complement)

    input_kmers, output_kmers, k_high = helper.extract_kmers(reads, input_length, spacing)
    input_seq, output_seq, input_stats_map = \
        helper.label_integer_encode_kmers(input_kmers, output_kmers)

    input_one_hot_cube = helper.encode(k_high, input_seq)
    output_one_hot_cube = helper.encode(bases_to_predict, output_seq)

    model = RandomPredictModel(bases_to_predict)

    start_time = time.time()
    model.fit(input_one_hot_cube, output_one_hot_cube)
    end_time = time.time()
    print("Fitting took " + str(end_time - start_time) + "s")

    helper.predict_and_validate(input_one_hot_cube, output_one_hot_cube, model)


if __name__ == "__main__":
    main()