import sys

sys.path.append('../../')

import time

import app.app_helper as helper
from predict.feedforward.RandomPredictModel import RandomPredictModel


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

    # TODO: maybe refactor this again and put extract_read_matrix into both feedforward_helper and rnn_helper which both delegate to generic_helper but tweak the parameters
    input_seq, input_quality, output_seq, shifted_output_seq, input_stats_map = helper.extract_read_matrix(paths,
                                                                                                           input_length,
                                                                                                           spacing,
                                                                                                           bases_to_predict,
                                                                                                           include_reverse_complement,
                                                                                                           unique,
                                                                                                           with_shifted_output=False)

    input_one_hot_cube, output_one_hot_cube, shifted_output_seq_cube = helper.encode_reads(input_length,
                                                                                           bases_to_predict,
                                                                                           input_seq,
                                                                                           input_quality,
                                                                                           output_seq,
                                                                                           shifted_output_seq,
                                                                                           has_quality=has_quality,
                                                                                           as_matrix=as_matrix)

    model = RandomPredictModel(bases_to_predict)

    start_time = time.time()
    model.fit(input_one_hot_cube, output_one_hot_cube)
    end_time = time.time()
    print("Fitting took " + str(end_time - start_time) + "s")

    helper.predict_and_validate(input_one_hot_cube, output_one_hot_cube, model, bases_to_predict,
                                             as_matrix=as_matrix)


if __name__ == "__main__":
    main()