import sys
sys.path.append('../../')

import numpy as np

import app.app_helper as helper
import app.rnn.app_helper as rnn_helper
from preprocess.KmerLabelEncoder import KmerLabelEncoder
from onehot.OneHotMatrix import OneHotMatrixEncoder
from predict.rnn.KerasLSTMModel import KerasLSTMModel

def main():
    input_length = 26
    prediction_length = 1
    has_quality = False

    label_encoder = KmerLabelEncoder()
    one_hot_encoder = OneHotMatrixEncoder(input_length)

    model = KerasLSTMModel(has_quality=has_quality, prediction_length=prediction_length, latent_dim=100)

    model.load_weights('../weights/my_model_weights.h5')

    while True:
        print("Enter a kmer of length " + str(input_length) + ": ")
        kmer = input().upper()
        validation_result = helper.validate_kmer(kmer, input_length)
        if validation_result is not None:
            print(validation_result)
            continue

        input_seq, input_quality, output_seq, shifted_output_seq = label_encoder.encode_kmers([kmer], [], [])
        print("Encoded kmer: " + str(input_seq))

        input_one_hot_cube = one_hot_encoder.encode_sequences(input_seq)
        print("One-Hot Encoded kmer: " + str(input_one_hot_cube))

        decoded_prediction = rnn_helper.predict(input_one_hot_cube, model, prediction_length)

        print("Predicted: " + np.array_str(decoded_prediction[0]) + " from " + kmer)


if __name__ == "__main__":
    main()
