import numpy as np

import constants.EncodingConstants as CONSTANTS
from onehot.OneHotVector import OneHotVectorDecoder
from preprocess.KmerLabelEncoder import KmerLabelEncoder

def predict_next_n_bases(model, seed, n):
    label_encoder = KmerLabelEncoder()
    prediction_length = 1
    one_hot_decoder = OneHotVectorDecoder(prediction_length, encoding_constants=CONSTANTS)

    seed_length = len(seed)

    basewise_probabilities = np.zeros((n, len(CONSTANTS.ONE_HOT_ENCODING)))

    remaining_length = n
    model.reset_states()
    current_sequence = str(seed)
    length = len(current_sequence)

    input_seq = label_encoder.encode_kmers([seed[:length-1]], [], with_shifted_output=False)[0]
    model.predict(input_seq)
    while remaining_length > 0:
        base = current_sequence[length-1:length]
        base_encoding = label_encoder.encode_kmers([base], [], with_shifted_output=False)[0]

        prediction = model.predict(base_encoding)
        decoded_prediction = one_hot_decoder.decode_sequences(prediction)
        current_sequence += decoded_prediction[0][0]
        basewise_probabilities[length - seed_length] = prediction[0]

        remaining_length -= 1
        length += 1
    return current_sequence, basewise_probabilities

def regenerate_sequence(min_seed_length, model, full_sequence):
    label_encoder = KmerLabelEncoder()
    prediction_length = 1
    one_hot_decoder = OneHotVectorDecoder(prediction_length, encoding_constants=CONSTANTS)

    sequence_length = len(full_sequence)
    start_string = full_sequence[0:min_seed_length]

    bases_to_predict = sequence_length - min_seed_length

    basewise_probabilities = np.zeros((bases_to_predict, len(CONSTANTS.ONE_HOT_ENCODING)))

    remaining_length = bases_to_predict
    model.reset_states()
    current_sequence = str(start_string)
    length = min_seed_length

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
    return current_sequence, basewise_probabilities