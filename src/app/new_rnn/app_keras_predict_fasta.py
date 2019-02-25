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
    min_seed_length = 26
    importer = SequenceImporter()
    reverser = SequenceReverser()

    path = '../data/ecoli_contigs/ecoli_contig_1000.fasta'
    static_offset = 0
    sequence = importer.import_fasta([path])[0]

    embedding_dim = 128
    latent_dim = 64

    model = SingleLSTMModel(min_seed_length=min_seed_length, stateful=True, batch_size=1, embedding_dim=embedding_dim,
                            latent_dim=latent_dim, with_gpu=True)

    model.load_weights('../weights/my_model_weights.h5')

    forward_predict = predict(model, min_seed_length, sequence, static_offset, directory="forward")
    reverse_predict = predict(model, min_seed_length, reverser.reverse_complement(sequence), static_offset, directory="reverse_complement")

    viz = SequenceRegenerationViz()
    viz.align_complements(forward_predict, reverse_predict, min_seed_length, static_offset)

def predict(model, implementation, min_seed_length, sequence, static_offset, directory=None):
    validator = SequenceMatchCalculator()
    viz = SequenceRegenerationViz(directory)
    label_encoder = KmerLabelEncoder()
    offset_sequence = sequence[static_offset:]

    predicted_string_with_seed, basewise_probabilities = helper.regenerate_sequence(implementation, min_seed_length, model, offset_sequence)

    predicted_sequence = predicted_string_with_seed[min_seed_length:]
    actual_sequence = offset_sequence[min_seed_length:]

    matches = validator.compare_sequences(predicted_sequence, actual_sequence)
    correct_index_vector = label_encoder.encode_kmers([actual_sequence], [], [])[0][0]

    viz.compare_sequences(sequence, predicted_string_with_seed, min_seed_length, matches, offset=static_offset)
    viz.top_base_probability_plot(basewise_probabilities, correct_index_vector, offset=min_seed_length+static_offset)
    viz.sliding_window_average_plot(matches, offset=min_seed_length+static_offset)

    top_base_probability = np.max(basewise_probabilities, axis=1)
    viz.sliding_window_average_plot(top_base_probability, offset=min_seed_length+static_offset, id="rnn_top_prediction_")
    return predicted_string_with_seed

if __name__ == "__main__":
    main()