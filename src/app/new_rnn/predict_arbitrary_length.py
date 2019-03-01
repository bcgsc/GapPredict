import sys
sys.path.append('../../')

from predict.new_rnn.SingleLSTMModel import SingleLSTMModel
from preprocess.SequenceImporter import SequenceImporter
from preprocess.SequenceReverser import SequenceReverser
from preprocess.SequenceMatchCalculator import SequenceMatchCalculator
from viz.SequenceRegenerationViz import SequenceRegenerationViz
from preprocess.KmerLabelEncoder import KmerLabelEncoder

import app.new_rnn.predict_helper as helper
import tensorflow as tf
import numpy as np

def main():
    weights_path = "../weights/"
    fasta_path = '../data/real_gaps/sealer_filled/7465348_13506-14596.fasta'
    id="7465348_13506-14596"
    embedding_dim = 128
    latent_dim = 256
    length_to_predict = 750
    predict_arbitrary_length(weights_path, id, fasta_path, embedding_dim, latent_dim, length_to_predict)

def predict_arbitrary_length(weights_path, id, fasta_path, embedding_dim, latent_dim, length_to_predict, base_path=None, gpu="0"):
    importer = SequenceImporter()
    reverser = SequenceReverser()

    sequences = importer.import_fasta([fasta_path])

    left_flank = sequences[0]
    right_flank = reverser.reverse_complement(sequences[1])
    reference = None

    model = SingleLSTMModel(min_seed_length=None, stateful=True, batch_size=1, embedding_dim=embedding_dim,
                            latent_dim=latent_dim, with_gpu=True)

    model.load_weights(weights_path + 'my_model_weights.h5')

    forward_predict = predict(model, left_flank, length_to_predict, reference, base_path=base_path, directory="predict_gap/forward", gpu=gpu)
    reverse_predict = predict(model, right_flank, length_to_predict, reference, base_path=base_path, directory="predict_gap/reverse_complement", gpu=gpu)

    viz = SequenceRegenerationViz(root_directory=base_path)
    viz.save_complements(forward_predict, reverse_predict, id, fasta_ref=fasta_path)

def predict(model, seed, prediction_length, reference, base_path=None, directory=None, gpu="0"):
    validator = SequenceMatchCalculator()
    label_encoder = KmerLabelEncoder()

    seed_length = len(seed)
    with tf.device('/gpu:' + gpu):
        predicted_string_with_seed, basewise_probabilities = helper.predict_next_n_bases(model, seed, prediction_length)

    viz = SequenceRegenerationViz(root_directory=base_path, directory=directory)
    viz.save_probabilities(basewise_probabilities)

    if reference:
        predicted_sequence = predicted_string_with_seed[seed_length:]
        actual_sequence = reference[seed_length:]

        matches = validator.compare_sequences(predicted_sequence, actual_sequence)
        correct_index_vector = label_encoder.encode_kmers([actual_sequence], [], [])[0][0]

        viz.compare_sequences(reference, predicted_string_with_seed, seed_length, matches)
        viz.top_base_probability_plot(basewise_probabilities, correct_index_vector, offset=seed_length)
        viz.sliding_window_average_plot(matches, offset=seed_length)

        top_base_probability = np.max(basewise_probabilities, axis=1)
        viz.sliding_window_average_plot(top_base_probability, offset=seed_length, id="rnn_top_prediction_")
    return predicted_string_with_seed

if __name__ == "__main__":
    main()