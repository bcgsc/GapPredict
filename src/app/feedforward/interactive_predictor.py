import sys
sys.path.append('../../')

import numpy as np

import app.app_helper as helper
from KmerLabelEncoder import KmerLabelEncoder
from onehot.OneHotVector import OneHotVectorEncoder, OneHotVectorDecoder
from predict.feedforward.KerasVanillaModel import KerasVanillaModel

def main():
    input_length = 26
    bases_to_predict = 1

    label_encoder = KmerLabelEncoder()
    one_hot_encoder = OneHotVectorEncoder(input_length)
    one_hot_decoder = OneHotVectorDecoder(bases_to_predict)

    model = KerasVanillaModel(input_length, bases_to_predict, batch_size=64, epochs=10)

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

        prediction = model.predict(input_one_hot_cube)
        decoded_prediction = one_hot_decoder.decode_sequences(prediction)

        print("Predicted: " + np.array_str(decoded_prediction[0]) + " from " + kmer)


if __name__ == "__main__":
    main()
