import sys
sys.path.append('../../')

import time

from sklearn import model_selection

import app.app_helper as helper
import app.rnn.app_helper as rnn_helper
from predict.rnn.KerasLSTMModel import KerasLSTMModel


def main():
    include_reverse_complement = True
    input_length = 26
    bases_to_predict = 1
    spacing = 0
    has_quality = False
    unique = False

    arguments = sys.argv[1:]
    paths = arguments if len(arguments) > 0 else ['../data/read_1_300.fastq']
    input_kmers, output_kmers, quality_vectors = helper.extract_kmers(paths, input_length, spacing, bases_to_predict, include_reverse_complement, unique)

    input_kmers_train, input_kmers_valid, output_kmers_train, output_kmers_valid, quality_train, quality_valid = model_selection.train_test_split(
        input_kmers, output_kmers, quality_vectors, test_size=0.01, random_state=123)

    print("Validation set")
    input_seq_valid, input_quality_valid, output_seq_valid, shifted_output_seq_valid, input_stats_map_valid = \
        helper.label_integer_encode_kmers(input_kmers_valid, output_kmers_valid, quality_valid, with_shifted_output=False)
    #hack to get a random shuffle of some size TODO: make this use numpy random so we can do 0% or 100% as well
    print("Encoding data set")
    input_one_hot_cube_valid = helper.encode(input_length, input_seq_valid, input_quality_valid, has_quality=has_quality)
    output_one_hot_cube_valid = helper.encode(bases_to_predict, output_seq_valid, None, has_quality=has_quality)

    model = KerasLSTMModel(has_quality=has_quality, prediction_length=bases_to_predict, batch_size=64, epochs=1, latent_dim=100)

    start_time = time.time()
    model.load_weights('../weights/my_model_weights.h5')
    end_time = time.time()
    print("Loading weights took " + str(end_time - start_time) + "s")

    print()
    print("Computing input statistics...")
    print("Unique mappings: " + str(input_stats_map_valid.get_total_unique_mappings_per_input()))
    print()
    print("Output stats: " + str(input_stats_map_valid.get_output_stats()))
    print()

    print("Predicting data set")
    rnn_helper.predict_and_validate(input_one_hot_cube_valid, output_one_hot_cube_valid, model, bases_to_predict)


if __name__ == "__main__":
    main()

