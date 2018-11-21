import sys
sys.path.append('../../')

import numpy as np

import app.rnn.app_helper as helper
from KmerLabelEncoder import KmerLabelEncoder
from constants import RnnEncodingConstants as CONSTANTS
from onehot.OneHotMatrix import OneHotMatrixEncoder
from predict.KerasLSTMModel import KerasLSTMModel


def validate_kmer(kmer, bases_to_predict):
    kmer_length = len(kmer)

    if kmer_length != bases_to_predict:
        return "Invalid kmer length"

    for l in range(kmer_length):
        if kmer[l] not in CONSTANTS.BASES:
            return "Invalid kmer base"

    return None

def main():
    input_length = 26
    bases_to_predict = 1
    has_quality = False
    fill_in_the_blanks = False

    label_encoder = KmerLabelEncoder()
    one_hot_encoder = OneHotMatrixEncoder(input_length)

    model = KerasLSTMModel(has_quality=has_quality, prediction_length=bases_to_predict, batch_size=64, epochs=1, latent_dim=100)

    model.load_weights('../weights/my_model_weights.h5')

    while True:
        print("Enter a kmer of length " + str(input_length) + ": ")
        kmer = input().upper()
        validation_result = validate_kmer(kmer, input_length)
        if validation_result is not None:
            print(validation_result)
            continue

        #TODO: the 2nd argument is a hack until we make encode_kmers robust to empty arrays, we ignore the output encodings so it doesn't matter
        input_seq, input_quality, output_seq, shifted_output_seq = label_encoder.encode_kmers([kmer], ["A"], [], fill_in_the_blanks=fill_in_the_blanks)
        print("Encoded kmer: " + str(input_seq))

        input_one_hot_cube = one_hot_encoder.encode_sequences(input_seq)
        print("One-Hot Encoded kmer: " + str(input_one_hot_cube))

        decoded_prediction = helper.predict(input_one_hot_cube, model, bases_to_predict)

        print("Predicted: " + np.array_str(decoded_prediction[0]) + " from " + kmer)


if __name__ == "__main__":
    main()
