import numpy as np

import app.new_rnn.implementation_constants as IMP_CONSTANTS
import constants.EncodingConstants as CONSTANTS
from onehot.OneHotVector import OneHotVectorDecoder
from preprocess.KmerLabelEncoder import KmerLabelEncoder

def _first_mismatch(seq1, seq2):
    length = min(len(seq1), len(seq2))
    for i in range(length):
        if seq1[i] != seq2[i]:
            return i
    else:
        return None

def regenerate_sequence_with_reseeding(implementation, min_seed_length, model, full_sequence):
    lower_bound = 0
    upper_bound = min_seed_length
    sequence_length = len(full_sequence)
    results = []

    while lower_bound + min_seed_length <= upper_bound:
        model.reset_states()
        sequence = full_sequence[lower_bound:sequence_length]
        predicted_sequence, basewise_probabilities = regenerate_sequence(implementation, min_seed_length, model, sequence)
        first_mismatch_idx = _first_mismatch(predicted_sequence, sequence) + lower_bound

        tuple = (predicted_sequence, sequence, basewise_probabilities, lower_bound, first_mismatch_idx)
        results.append(tuple)

        if first_mismatch_idx is None:
            break
        else:
            lower_bound = int((lower_bound + first_mismatch_idx)/2)
            upper_bound = first_mismatch_idx

    return results

#TODO: combine this with the general function later on if we still want this
def regenerate_sequence_with_spacing(min_seed_length, model, full_sequence, spacing):
    label_encoder = KmerLabelEncoder()
    prediction_length = 1
    one_hot_decoder = OneHotVectorDecoder(prediction_length, encoding_constants=CONSTANTS)

    sequence_length = len(full_sequence)
    seed_length = min_seed_length + spacing
    start_string = full_sequence[0:seed_length]

    bases_to_predict = sequence_length - seed_length

    basewise_probabilities = np.zeros((bases_to_predict, len(CONSTANTS.ONE_HOT_ENCODING)))

    remaining_length = bases_to_predict
    current_sequence = str(start_string)
    seed_length = min_seed_length

    seed = current_sequence[0:seed_length - 1]
    input_seq = label_encoder.encode_kmers([seed], [], with_shifted_output=False)[0]
    model.predict(input_seq)
    while remaining_length > 0:
        base = current_sequence[seed_length - 1:seed_length]
        base_encoding = label_encoder.encode_kmers([base], [], with_shifted_output=False)[0]

        prediction = model.predict(base_encoding)
        decoded_prediction = one_hot_decoder.decode_sequences(prediction)
        current_sequence += decoded_prediction[0][0]
        basewise_probabilities[seed_length - min_seed_length] = prediction[0]

        remaining_length -= 1
        seed_length += 1
    return current_sequence, basewise_probabilities

def regenerate_sequence(implementation, min_seed_length, model, full_sequence):
    label_encoder = KmerLabelEncoder()
    prediction_length = 1
    one_hot_decoder = OneHotVectorDecoder(prediction_length, encoding_constants=CONSTANTS)

    sequence_length = len(full_sequence)
    start_string = full_sequence[0:min_seed_length]

    bases_to_predict = sequence_length - min_seed_length

    basewise_probabilities = np.zeros((bases_to_predict, len(CONSTANTS.ONE_HOT_ENCODING)))

    remaining_length = bases_to_predict
    if implementation == IMP_CONSTANTS.STATIC_PREDICTION:
        current_sequence = str(start_string)
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
    elif implementation == IMP_CONSTANTS.SINGLE_BASE_PREDICTION:
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
    elif implementation == IMP_CONSTANTS.EXTENDING_SEQUENCE_PREDICTION:
        current_sequence = str(start_string)
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
    return current_sequence, basewise_probabilities

def regenerate_sequence_second_choice(implementation, min_seed_length, model, sampler_model, full_sequence):
    reverse_encoding = CONSTANTS.REVERSE_INTEGER_ENCODING
    label_encoder = KmerLabelEncoder()

    sequence_length = len(full_sequence)
    start_string = full_sequence[0:min_seed_length]

    bases_to_predict = sequence_length - min_seed_length

    basewise_probabilities = np.zeros((bases_to_predict, len(CONSTANTS.ONE_HOT_ENCODING)))

    remaining_length = bases_to_predict
    if implementation == IMP_CONSTANTS.SINGLE_BASE_PREDICTION:
        current_sequence = str(start_string)
        length = min_seed_length

        seed = current_sequence[0:length - 1]
        input_seq = label_encoder.encode_kmers([seed], [], with_shifted_output=False)[0]
        model.predict(input_seq)
        while remaining_length > 0:
            base = current_sequence[length-1:length]
            base_encoding = label_encoder.encode_kmers([base], [], with_shifted_output=False)[0]

            prediction = model.predict(base_encoding)[0]
            best_prediction = reverse_encoding[_nth_largest_idx(prediction, 1)]
            second_best_prediction = reverse_encoding[_nth_largest_idx(prediction, 2)]
            chosen_prediction = _branch_and_choose(best_prediction, second_best_prediction, current_sequence, sampler_model)
            current_sequence += chosen_prediction
            basewise_probabilities[length - min_seed_length] = prediction

            remaining_length -= 1
            length += 1
    return current_sequence, basewise_probabilities

def _nth_largest_idx(array, n):
    copy = np.copy(array)
    for i in range(n-1):
        largest_idx = np.argmax(copy)
        copy[largest_idx] = 0
    return np.argmax(copy)

def _branch_and_choose(best_prediction, second_best_prediction, current_sequence, model):
    label_encoder = KmerLabelEncoder()
    best_seed = str(current_sequence) + best_prediction
    second_best_seed = str(current_sequence) + second_best_prediction

    encoded_best_seed = label_encoder.encode_kmers([best_seed], [], with_shifted_output=False)[0]
    encoded_second_best_seed = label_encoder.encode_kmers([second_best_seed], [], with_shifted_output=False)[0]

    best_seed_prediction = model.predict(encoded_best_seed)[0]
    second_best_seed_prediction = model.predict(encoded_second_best_seed)[0]

    if np.max(best_seed_prediction) > np.max(second_best_seed_prediction):
        return best_prediction
    else:
        return second_best_prediction