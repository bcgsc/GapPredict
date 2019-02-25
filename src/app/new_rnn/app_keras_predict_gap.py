import sys
sys.path.append('../../')

from predict.new_rnn.SingleLSTMModel import SingleLSTMModel
from preprocess.SequenceImporter import SequenceImporter
from preprocess.SequenceReverser import SequenceReverser
from preprocess.SequenceMatchCalculator import SequenceMatchCalculator
from viz.SequenceRegenerationViz import SequenceRegenerationViz
from preprocess.KmerLabelEncoder import KmerLabelEncoder

import app.new_rnn.predict_helper as helper

import numpy as np


def main():
    importer = SequenceImporter()
    reverser = SequenceReverser()

    path = '../data/ecoli_contigs/ecoli_contig_1000.fasta'
    sequence = importer.import_fasta([path])[0]

    seed_length = 400
    prediction_length=len(sequence)-seed_length
    left_flank = sequence[:seed_length]
    right_flank = reverser.reverse_complement(sequence[prediction_length:])

    embedding_dim = 128
    latent_dim = 64

    model = SingleLSTMModel(min_seed_length=None, stateful=True, batch_size=1, embedding_dim=embedding_dim,
                            latent_dim=latent_dim, with_gpu=True)

    model.load_weights('../weights/my_model_weights.h5')

    forward_predict = predict(model, left_flank, prediction_length, sequence, directory="forward")
    reverse_predict = predict(model, right_flank, prediction_length, reverser.reverse_complement(sequence), directory="reverse_complement")

    viz = SequenceRegenerationViz()
    viz.align_complements(forward_predict, reverse_predict, seed_length)

def predict(model, seed, prediction_length, reference, directory=None):
    validator = SequenceMatchCalculator()
    viz = SequenceRegenerationViz(directory)
    label_encoder = KmerLabelEncoder()

    seed_length = len(seed)
    predicted_string_with_seed, basewise_probabilities = helper.predict_next_n_bases(model, seed, prediction_length)

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