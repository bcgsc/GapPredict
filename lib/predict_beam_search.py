from lstm.GapPredictModel import GapPredictModel
from preprocess.SequenceImporter import SequenceImporter
from preprocess.SequenceReverser import SequenceReverser
from utils.DataWriter import DataWriter
import utils.directory_utils as UTILS
from preprocess.KmerLabelDecoder import KmerLabelDecoder
from predict.BeamSearchPredictor import BeamSearchPredictor

import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

def predict_arbitrary_length(weights_path, gap_id, fasta_path, embedding_dim, latent_dim, length_to_predict, beam_length, base_path=None):
    importer = SequenceImporter()
    reverser = SequenceReverser()

    sequences = importer.import_fasta([fasta_path])

    left_flank = sequences[0]
    right_flank = reverser.reverse_complement(sequences[1])

    model = GapPredictModel(min_seed_length=None, stateful=False, embedding_dim=embedding_dim,
                            latent_dim=latent_dim, with_gpu=True)

    model.load_weights(weights_path + 'my_model_weights.h5')
    terminal_directory_character = UTILS.get_terminal_directory_character()
    directory = "beam_search" + terminal_directory_character + "predict_gap" + terminal_directory_character

    flank_id = gap_id + "_" + "left_forward"
    predict(model, left_flank, length_to_predict, beam_length, flank_id, base_path=base_path, directory=directory + "forward")
    flank_id = gap_id + "_" + "right_reverse_complement"
    predict(model, right_flank, length_to_predict, beam_length, flank_id, base_path=base_path, directory=directory + "reverse_complement")

def predict_reference(weights_path, gap_id, fasta_path, embedding_dim, latent_dim, min_seed_length, beam_length, base_path=None):
    importer = SequenceImporter()
    reverser = SequenceReverser()

    sequences = importer.import_fasta([fasta_path])[0:2]

    model = GapPredictModel(min_seed_length=min_seed_length, stateful=False, embedding_dim=embedding_dim,
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
        predict(model, forward_seed, prediction_length, beam_length, flank_id, base_path=base_path, directory=static_path + directionality)
        directionality = "reverse_complement"
        flank_id = gap_id + "_" + flank + "_" + directionality
        predict(model, reverse_complement_seed, prediction_length, beam_length, flank_id, base_path=base_path, directory=static_path + directionality)

def predict(model, seed, prediction_length, beam_length, flank_id, base_path=None, directory=None):
    predictor = BeamSearchPredictor(model)
    predicted_strings_with_seed, lg_sum_probabilities = predictor.predict_next_n_bases(seed, prediction_length, beam_length)
    writer = DataWriter(root_directory=base_path, directory=directory)
    writer.save_probabilities(lg_sum_probabilities, file_id="beam_search")
    decoder = KmerLabelDecoder()

    predictions_as_strings = []
    for i in range(len(predicted_strings_with_seed)):
        string = decoder.decode(predicted_strings_with_seed[i])
        predictions_as_strings.append(string)
    writer.write_beam_search_results(predictions_as_strings, flank_id)