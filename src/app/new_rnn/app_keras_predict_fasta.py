import sys
sys.path.append('../../')

from onehot.OneHotVector import OneHotVectorDecoder
from predict.new_rnn.SingleLSTMModel import SingleLSTMModel
from preprocess.SequenceImporter import SequenceImporter
from preprocess.SequenceMatchCalculator import SequenceMatchCalculator
import constants.EncodingConstants as CONSTANTS
from preprocess.KmerLabelEncoder import KmerLabelEncoder
from viz.SequenceRegenerationViz import SequenceRegenerationViz

import numpy as np

def main():
    min_seed_length = 26
    importer = SequenceImporter()
    validator = SequenceMatchCalculator()
    label_encoder = KmerLabelEncoder()
    viz = SequenceRegenerationViz()

    path = '../data/ecoli_contigs/ecoli_contig_1000.fasta'
    sequence = importer.import_fasta([path])[0]

    sequence_length = len(sequence)

    output_length = sequence_length - min_seed_length

    input = sequence[0:min_seed_length]
    bases_to_predict = sequence[min_seed_length:sequence_length]

    prediction_length = 1
    one_hot_decoder = OneHotVectorDecoder(prediction_length, encoding_constants=CONSTANTS)
    embedding_dim = 25
    latent_dim = 100

    implementation = 2

    basewise_probabilities = np.zeros((len(bases_to_predict), len(CONSTANTS.ONE_HOT_ENCODING)))

    if implementation == 1:
        model = SingleLSTMModel(min_seed_length=min_seed_length, stateful=False, embedding_dim=embedding_dim, latent_dim=latent_dim,
                                with_gpu=True)
        model.load_weights('../weights/my_model_weights.h5')

        remaining_length = output_length

        current_sequence = str(input)
        lower_bound = 0
        upper_bound = lower_bound + min_seed_length
        while remaining_length > 0:
            seed = current_sequence[lower_bound:upper_bound]
            input_seq = label_encoder.encode_kmers([seed], [], with_shifted_output=False)[0]

            prediction = model.predict(input_seq)
            decoded_prediction = one_hot_decoder.decode_sequences(prediction)
            current_sequence += decoded_prediction[0][0]
            basewise_probabilities[lower_bound] = prediction[0]

            remaining_length -= 1
            lower_bound += 1
            upper_bound += 1
    elif implementation == 2:
        model = SingleLSTMModel(min_seed_length=min_seed_length, stateful=True, batch_size=1, embedding_dim=embedding_dim, latent_dim=latent_dim,
                                with_gpu=True)
        model.load_weights('../weights/my_model_weights.h5')

        remaining_length = output_length

        current_sequence = str(input)
        length = min_seed_length

        #prep state
        seed = current_sequence[0:length - 1]
        input_seq = label_encoder.encode_kmers([seed], [], with_shifted_output=False)[0]
        model.predict(input_seq)
        while remaining_length > 0:
            base = current_sequence[length-1:length]
            base_encoding = label_encoder.encode_kmers([base], [], with_shifted_output=False)[0]

            prediction = model.predict(base_encoding)
            decoded_prediction = one_hot_decoder.decode_sequences(prediction)
            current_sequence += decoded_prediction[0][0]
            basewise_probabilities[length - min_seed_length] = prediction[0]

            remaining_length -= 1
            length += 1
    elif implementation == 3:
        model = SingleLSTMModel(min_seed_length=min_seed_length, embedding_dim=embedding_dim, latent_dim=latent_dim,
                                with_gpu=True)
        model.load_weights('../weights/my_model_weights.h5')

        remaining_length = output_length

        current_sequence = str(input)
        length = min_seed_length
        while remaining_length > 0:
            seed = current_sequence[0:length]
            input_seq = label_encoder.encode_kmers([seed], [], with_shifted_output=False)[0]

            prediction = model.predict(input_seq)
            decoded_prediction = one_hot_decoder.decode_sequences(prediction)
            current_sequence += decoded_prediction[0][0]
            basewise_probabilities[length - min_seed_length] = prediction[0]

            remaining_length -= 1
            length += 1

    predicted_string = current_sequence
    predicted_sequence = current_sequence[min_seed_length:sequence_length]
    matches = validator.compare_sequences(predicted_sequence, bases_to_predict)
    mean_match = np.mean(matches)
    print("Mean Match: " + str(mean_match))

    correct_index_vector = label_encoder.encode_kmers([bases_to_predict], [], [])[0][0]

    viz.compare_sequences(sequence, predicted_string, min_seed_length)
    viz.top_base_probability_plot(basewise_probabilities, correct_index_vector)
    viz.sliding_window_average_plot(matches)


if __name__ == "__main__":
    main()