import sys
sys.path.append('../../')

from predict.new_rnn.SingleLSTMModel import SingleLSTMModel
from preprocess.SequenceImporter import SequenceImporter
from preprocess.SequenceReverser import SequenceReverser
from viz.SequenceRegenerationViz import SequenceRegenerationViz
import constants.EncodingConstants as CONSTANTS
import utils.directory_utils as UTILS
import app.new_rnn.predict_helper as helper

import numpy as np
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def predict_arbitrary_length(weights_path, gap_id, fasta_path, embedding_dim, latent_dim, length_to_predict, prune_length, base_path=None):
    importer = SequenceImporter()
    reverser = SequenceReverser()

    sequences = importer.import_fasta([fasta_path])

    left_flank = sequences[0]
    right_flank = reverser.reverse_complement(sequences[1])

    model = SingleLSTMModel(min_seed_length=None, stateful=False, embedding_dim=embedding_dim,
                            latent_dim=latent_dim, with_gpu=True)

    model.load_weights(weights_path + 'my_model_weights.h5')
    terminal_directory_character = UTILS.get_terminal_directory_character()
    directory = "beam_search" + terminal_directory_character + "predict_gap" + terminal_directory_character

    flank_id = gap_id + "_" + "left_forward"
    predict(model, left_flank, length_to_predict, prune_length, flank_id, base_path=base_path, directory=directory + "forward")
    flank_id = gap_id + "_" + "right_reverse_complement"
    predict(model, right_flank, length_to_predict, prune_length, flank_id, base_path=base_path, directory=directory + "reverse_complement")

def predict_reference(weights_path, fasta_path, embedding_dim, latent_dim, min_seed_length, gap_id, prune_length, base_path=None):
    importer = SequenceImporter()
    reverser = SequenceReverser()

    sequences = importer.import_fasta([fasta_path])[0:2]

    model = SingleLSTMModel(min_seed_length=min_seed_length, stateful=False, embedding_dim=embedding_dim,
                            latent_dim=latent_dim, with_gpu=True)

    model.load_weights(weights_path + 'my_model_weights.h5')

    terminal_directory_character = UTILS.get_terminal_directory_character()
    first_directory = "beam_search" + terminal_directory_character + "regenerate_seq" + terminal_directory_character
    for i in range(len(sequences)):
        sequence = sequences[i]

        #TODO: messy since we assume that i will always be [0, 1]
        if i == 0:
            flank = "left_flank"
            static_path = first_directory + flank + terminal_directory_character
        elif i == 1:
            flank = "right_flank"
            static_path = first_directory + flank + terminal_directory_character

        forward_seed = sequence[:min_seed_length]
        prediction_length = len(sequence) - min_seed_length
        reverse_complement_seed = reverser.reverse_complement(sequence)[:min_seed_length]
        directionality = "forward"
        flank_id = gap_id + "_" + flank + "_" + directionality
        predict(model, forward_seed, prediction_length, prune_length, flank_id, base_path=base_path, directory=static_path + directionality)
        directionality = "reverse_complement"
        flank_id = gap_id + "_" + flank + "_" + directionality
        predict(model,reverse_complement_seed, prediction_length, prune_length, flank_id, base_path=base_path, directory=static_path + directionality)

def predict(model, seed, prediction_length, prune_length, flank_id, base_path=None, directory=None):
    predicted_strings_with_seed, lg_sum_probabilities = helper.predict_next_n_bases_beam_search(model, seed, prediction_length, prune_length)
    viz = SequenceRegenerationViz(root_directory=base_path, directory=directory)
    viz.save_probabilities(lg_sum_probabilities, fig_id="beam_search")

    predictions_as_strings = []
    for i in range(len(predicted_strings_with_seed)):
        string = "".join(CONSTANTS.REVERSE_INTEGER_ENCODING[predicted_strings_with_seed[i].astype(int)]) #TODO would be nice to have this as a decoder class
        predictions_as_strings.append(string)
    viz.write_beam_search_results(predictions_as_strings, flank_id)

def main():
    weights_path = "../weights/"
    fasta_path = '../data/real_gaps/sealer_filled/7465348_13506-14596.fasta'
    gap_id="7465348_13506-14596"
    embedding_dim = 128
    latent_dim = 512
    length_to_predict = 750
    prune_length = 10
    min_seed_length = 52
    predict_reference(weights_path, fasta_path, embedding_dim, latent_dim, min_seed_length, gap_id, prune_length)
    predict_arbitrary_length(weights_path, gap_id, fasta_path, embedding_dim, latent_dim, length_to_predict, prune_length)

if __name__ == "__main__":
    main()