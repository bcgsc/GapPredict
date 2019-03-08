import sys
sys.path.append('../../')

from predict.new_rnn.SingleLSTMModel import SingleLSTMModel
from preprocess.SequenceImporter import SequenceImporter
from preprocess.SequenceReverser import SequenceReverser
from preprocess.SequenceMatchCalculator import SequenceMatchCalculator
from viz.SequenceRegenerationViz import SequenceRegenerationViz
from preprocess.KmerLabelEncoder import KmerLabelEncoder
import utils.directory_utils as UTILS
import tensorflow as tf
import app.new_rnn.predict_helper as helper

import numpy as np

def predict_reference(weights_path, fasta_path, embedding_dim, latent_dim, min_seed_length, id, plots=False, base_path=None, gpu="0"):
    importer = SequenceImporter()
    reverser = SequenceReverser()

    sequences = importer.import_fasta([fasta_path])[0:2]

    model = SingleLSTMModel(min_seed_length=min_seed_length, stateful=True, batch_size=1, embedding_dim=embedding_dim,
                            latent_dim=latent_dim, with_gpu=True)

    model.load_weights(weights_path + 'my_model_weights.h5')

    terminal_directory_character = UTILS.get_terminal_directory_character()
    first_directory = "regenerate_seq" + terminal_directory_character
    forward_left_flank = None
    rc_left_flank = None
    forward_right_flank = None
    rc_right_flank = None
    for i in range(len(sequences)):
        sequence = sequences[i]

        #TODO: messy
        if i == 0:
            static_path = first_directory + "left_flank" + terminal_directory_character
        elif i == 1:
            static_path = first_directory + "right_flank" + terminal_directory_character

        viz = SequenceRegenerationViz(root_directory=base_path, directory=static_path)

        forward_predict = predict(model, min_seed_length, sequence, base_path=base_path, directory=static_path + "forward", plots=plots, gpu=gpu)
        reverse_predict = predict(model, min_seed_length, reverser.reverse_complement(sequence), base_path=base_path, directory=static_path + "reverse_complement", plots=plots, gpu=gpu)

        if i == 0:
            forward_left_flank = forward_predict
            rc_left_flank = reverse_predict
        elif i == 1:
            forward_right_flank = forward_predict
            rc_right_flank = reverse_predict

        viz.align_complements(forward_predict, reverse_predict, min_seed_length)

    viz = SequenceRegenerationViz(root_directory=base_path, directory=first_directory)
    viz.write_flank_predict_fasta(forward_left_flank, rc_left_flank, forward_right_flank, rc_right_flank, latent_dim, id)

def predict(model, min_seed_length, sequence, base_path=None, directory=None, plots=False, gpu="0"):
    validator = SequenceMatchCalculator()
    viz = SequenceRegenerationViz(root_directory=base_path, directory=directory)
    label_encoder = KmerLabelEncoder()

    with tf.device('/gpu:' + gpu):
        predicted_string_with_seed, basewise_probabilities = helper.regenerate_sequence(min_seed_length, model, sequence)

    predicted_sequence = predicted_string_with_seed[min_seed_length:]
    actual_sequence = sequence[min_seed_length:]

    matches = validator.compare_sequences(predicted_sequence, actual_sequence)
    correct_index_vector = label_encoder.encode_kmers([actual_sequence], [], [])[0][0]

    viz.compare_sequences(sequence, predicted_string_with_seed, min_seed_length, matches)
    viz.save_probabilities(basewise_probabilities)

    if plots:
        viz.sliding_window_average_plot(matches, offset=min_seed_length)
        viz.top_base_probability_plot(basewise_probabilities, correct_index_vector, offset=min_seed_length)
        top_base_probability = np.max(basewise_probabilities, axis=1)
        viz.sliding_window_average_plot(top_base_probability, offset=min_seed_length, id="rnn_top_prediction_")
    return predicted_string_with_seed


def main():
    weights_path = "../weights/"
    fasta_path = '../data/real_gaps/sealer_filled/7465348_13506-14596.fasta'
    embedding_dim = 128
    latent_dim = 256
    min_seed_length = 26
    plots = True
    id="7465348_13506-14596"

    predict_reference(weights_path, fasta_path, embedding_dim, latent_dim, min_seed_length, id, plots=plots)

if __name__ == "__main__":
    main()