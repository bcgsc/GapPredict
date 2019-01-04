import sys
sys.path.append('../../')

from preprocess.KmerLabelEncoder import KmerLabelEncoder
from onehot.OneHotVector import OneHotVectorEncoder, OneHotVectorDecoder
from predict.feedforward.KerasVanillaModel import KerasVanillaModel
from preprocess.SequenceImporter import SequenceImporter
from preprocess.SequenceMatchCalculator import SequenceMatchCalculator

import numpy as np

def main():
    input_length = 26
    importer = SequenceImporter()
    validator = SequenceMatchCalculator()
    label_encoder = KmerLabelEncoder()
    has_quality = False

    path = '../data/ecoli_contigs/ecoli_contig_1000.fasta'
    sequence = importer.import_fasta([path])[0].sequence

    sequence_length = len(sequence)

    output_length = sequence_length - input_length

    input = sequence[0:input_length]
    bases_to_predict = sequence[input_length:sequence_length]

    prediction_length = 1
    remaining_length = output_length

    one_hot_encoder = OneHotVectorEncoder(input_length)
    one_hot_decoder = OneHotVectorDecoder(prediction_length)

    model = KerasVanillaModel(input_length, prediction_length)
    model.load_weights('../weights/my_model_weights.h5')

    current_sequence = str(input)
    lower_bound = 0
    upper_bound = lower_bound + input_length
    while remaining_length > 0:
        seed = current_sequence[lower_bound:upper_bound]
        input_seq, input_quality, output_seq, shifted_output_seq = label_encoder.encode_kmers([seed], [], [])

        input_one_hot_cube = one_hot_encoder.encode_sequences(input_seq)

        prediction = model.predict(input_one_hot_cube)
        decoded_prediction = one_hot_decoder.decode_sequences(prediction)
        current_sequence += decoded_prediction[0][0]

        remaining_length -= 1
        lower_bound += 1
        upper_bound += 1

    predicted_sequence = current_sequence[input_length:sequence_length]
    print("Predicted: " + predicted_sequence + " from " + input)
    matches = validator.compare_sequences(predicted_sequence, bases_to_predict)
    mean_match = np.mean(matches)
    print("Matches: " + str(matches))
    print("Mean Match: " + str(mean_match))


if __name__ == "__main__":
    main()