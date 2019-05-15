import sys
sys.path.append('../../')

from predict.SingleLSTMModel import SingleLSTMModel
from preprocess.SequenceImporter import SequenceImporter
from preprocess.SequenceReverser import SequenceReverser
from utils.DataWriter import DataWriter
import utils.directory_utils as UTILS
import app.new_rnn.predict_helper as helper

def predict_arbitrary_length(weights_path, gap_id, fasta_path, embedding_dim, latent_dim, length_to_predict, base_path=None):
    importer = SequenceImporter()
    reverser = SequenceReverser()

    sequences = importer.import_fasta([fasta_path])

    left_flank = sequences[0]
    right_flank = reverser.reverse_complement(sequences[1])
    reference = None

    model = SingleLSTMModel(min_seed_length=None, stateful=True, batch_size=1, embedding_dim=embedding_dim,
                            latent_dim=latent_dim, with_gpu=True)

    model.load_weights(weights_path + 'my_model_weights.h5')

    forward_predict = predict(model, left_flank, length_to_predict, reference, base_path=base_path, directory="predict_gap/forward")
    reverse_predict = predict(model, right_flank, length_to_predict, reference, base_path=base_path, directory="predict_gap/reverse_complement")

    viz = DataWriter(root_directory=base_path)
    postfix = "_LD_"+str(latent_dim)
    viz.save_complements(forward_predict, reverse_predict, gap_id, postfix=postfix, fasta_ref=fasta_path)

def predict_reference(weights_path, gap_id, fasta_path, embedding_dim, latent_dim, min_seed_length, plots=False, base_path=None):
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

        #TODO: messy since we assume that i will always be [0, 1]
        if i == 0:
            static_path = first_directory + "left_flank" + terminal_directory_character
        elif i == 1:
            static_path = first_directory + "right_flank" + terminal_directory_character

        viz = DataWriter(root_directory=base_path, directory=static_path)

        forward_predict = predict(model, min_seed_length, sequence, base_path=base_path, directory=static_path + "forward", plots=plots)
        reverse_predict = predict(model, min_seed_length, reverser.reverse_complement(sequence), base_path=base_path, directory=static_path + "reverse_complement", plots=plots)

        if i == 0:
            forward_left_flank = forward_predict
            rc_left_flank = reverse_predict
        elif i == 1:
            forward_right_flank = forward_predict
            rc_right_flank = reverse_predict

        viz.align_complements(forward_predict, reverse_predict, min_seed_length)

    viz = DataWriter(root_directory=base_path, directory=first_directory)
    viz.write_flank_predict_fasta(forward_left_flank, rc_left_flank, forward_right_flank, rc_right_flank, latent_dim, gap_id)

def predict(model, min_seed_length, sequence, base_path=None, directory=None):
    viz = DataWriter(root_directory=base_path, directory=directory)

    predicted_string_with_seed, basewise_probabilities = helper.regenerate_sequence(min_seed_length, model, sequence)
    predicted_string_greedy_with_seed, basewise_probabilities_greedy = helper.regenerate_sequence(min_seed_length, model, sequence, use_reference_to_seed=False)
    predicted_string_random_with_seed, random_probability_vector = helper.regenerate_sequence_randomly(min_seed_length, model, sequence)

    viz.save_probabilities(basewise_probabilities)
    viz.save_probabilities(basewise_probabilities_greedy, fig_id="greedy")
    viz.save_probabilities(random_probability_vector, fig_id="random")
    return predicted_string_with_seed