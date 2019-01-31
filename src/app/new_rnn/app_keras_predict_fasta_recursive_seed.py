import sys
sys.path.append('../../')

from predict.new_rnn.SingleLSTMModel import SingleLSTMModel
from preprocess.SequenceImporter import SequenceImporter
from preprocess.SequenceMatchCalculator import SequenceMatchCalculator
from viz.SequenceRegenerationViz import SequenceRegenerationViz
from preprocess.KmerLabelEncoder import KmerLabelEncoder

import app.new_rnn.predict_helper as helper
import app.new_rnn.implementation_constants as IMP_CONSTANTS
import numpy as np

def main():
    min_seed_length = 26
    importer = SequenceImporter()
    validator = SequenceMatchCalculator()
    viz = SequenceRegenerationViz()
    label_encoder = KmerLabelEncoder()

    path = '../data/ecoli_contigs/ecoli_contig_1000.fasta'
    static_offset = 0
    sequence = importer.import_fasta([path])[0]
    sequence_length = len(sequence)

    offset_sequence = sequence[static_offset:sequence_length]

    embedding_dim = 25
    latent_dim = 100

    implementation = IMP_CONSTANTS.SINGLE_BASE_PREDICTION
    if implementation == IMP_CONSTANTS.STATIC_PREDICTION:
        model = SingleLSTMModel(min_seed_length=min_seed_length, stateful=False, embedding_dim=embedding_dim,
                                latent_dim=latent_dim,
                                with_gpu=True)
    if implementation == IMP_CONSTANTS.SINGLE_BASE_PREDICTION:
        model = SingleLSTMModel(min_seed_length=min_seed_length, stateful=True, batch_size=1, embedding_dim=embedding_dim, latent_dim=latent_dim,
                                with_gpu=True)
    elif implementation == IMP_CONSTANTS.EXTENDING_SEQUENCE_PREDICTION:
        model = SingleLSTMModel(min_seed_length=min_seed_length, embedding_dim=embedding_dim, latent_dim=latent_dim,
                                with_gpu=True)
    model.load_weights('../weights/my_model_weights.h5')
    results = helper.regenerate_sequence_with_reseeding(implementation, min_seed_length, model, offset_sequence)

    alignment_data = []

    for i in range(len(results)):
        tuple = results[i]
        predicted_sequence = tuple[0]
        reference_sequence = tuple[1]
        basewise_probabilities = tuple[2]
        current_lower_bound = tuple[3]

        seedless_predicted_sequence = predicted_sequence[min_seed_length:]
        seedless_reference_sequence = reference_sequence[min_seed_length:]

        matches = validator.compare_sequences(seedless_predicted_sequence, seedless_reference_sequence)

        correct_index_vector = label_encoder.encode_kmers([seedless_reference_sequence], [], [])[0][0]

        viz.top_base_probability_plot(basewise_probabilities, correct_index_vector, offset=static_offset+current_lower_bound+min_seed_length, id=str(i)+"_")
        viz.sliding_window_average_plot(matches, offset=static_offset+current_lower_bound+min_seed_length, id=str(i)+"_")

        top_base_probability = np.max(basewise_probabilities, axis=1)
        viz.sliding_window_average_plot(top_base_probability, offset=static_offset+current_lower_bound+min_seed_length, id=str(i)+"_rnn_top_prediction_")

        alignment_tuple = (predicted_sequence, matches, current_lower_bound)
        alignment_data.append(alignment_tuple)

    viz.compare_multiple_sequences(sequence, alignment_data, min_seed_length, static_offset)


if __name__ == "__main__":
    main()