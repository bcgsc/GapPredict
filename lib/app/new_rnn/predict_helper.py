import numpy as np

import constants.EncodingConstants as CONSTANTS
from onehot.OneHotVector import OneHotVectorDecoder
from preprocess.KmerLabelEncoder import KmerLabelEncoder

def predict_next_n_bases_beam_search(model, seed, prediction_length, prune_length):
    label_encoder = KmerLabelEncoder()
    seed_length = len(seed)
    total_prediction_length = seed_length + prediction_length

    current_length = seed_length
    encoded_seed = label_encoder.encode_kmers([seed], [], with_shifted_output=False)[0]
    tree_breadth = 1
    current_level = (encoded_seed, np.array([0]))
    while current_length < total_prediction_length:
        seed_vectors = current_level[0]
        lg_sum_p = current_level[1]
        predictions = model.predict(seed_vectors)

        current_length += 1

        level_expansion_factor = predictions.shape[1]
        tree_breadth = tree_breadth * level_expansion_factor
        next_level_seeds = np.zeros((tree_breadth, current_length))
        next_level_lg_sum_p = np.zeros(tree_breadth)
        for i in range(len(predictions)):
            prediction = predictions[i]
            original_seed = seed_vectors[i]
            original_lg_sum_p = lg_sum_p[i]
            for j in range(level_expansion_factor):
                probability = prediction[j]
                row = i * level_expansion_factor + j
                next_level_seeds[row, 0:len(original_seed)] = original_seed[:]
                next_level_seeds[row, -1] = j
                next_level_lg_sum_p[row] = original_lg_sum_p + np.log(probability)
        tree_breadth = min(tree_breadth, prune_length)

        sorted_idx = np.argsort(next_level_lg_sum_p)[::-1]
        sorted_next_level_seeds = next_level_seeds[sorted_idx]
        sorted_next_level_lg_sum_p = next_level_lg_sum_p[sorted_idx]
        current_level = (sorted_next_level_seeds[:tree_breadth], sorted_next_level_lg_sum_p[:tree_breadth])
    return current_level[0], current_level[1]

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

def regenerate_sequence(min_seed_length, model, full_sequence, use_reference_to_seed=True):
    label_encoder = KmerLabelEncoder()
    prediction_length = 1
    one_hot_decoder = OneHotVectorDecoder(prediction_length, encoding_constants=CONSTANTS)

    sequence_length = len(full_sequence)

    bases_to_predict = sequence_length - min_seed_length

    basewise_probabilities = np.zeros((bases_to_predict, len(CONSTANTS.ONE_HOT_ENCODING)))

    remaining_length = bases_to_predict
    model.reset_states()
    current_sequence = str(full_sequence[0:min_seed_length])
    length = min_seed_length

    seed = full_sequence[0:length - 1]
    input_seq = label_encoder.encode_kmers([seed], [], with_shifted_output=False)[0]
    model.predict(input_seq)
    while remaining_length > 0:
        base = full_sequence[length-1:length] if use_reference_to_seed else current_sequence[length-1:length]
        base_encoding = label_encoder.encode_kmers([base], [], with_shifted_output=False)[0]

        prediction = model.predict(base_encoding)
        decoded_prediction = one_hot_decoder.decode_sequences(prediction)
        current_sequence += decoded_prediction[0][0]
        basewise_probabilities[length - min_seed_length] = prediction[0]

        remaining_length -= 1
        length += 1
    return current_sequence, basewise_probabilities

def regenerate_sequence_randomly(min_seed_length, model, full_sequence):
    label_encoder = KmerLabelEncoder()

    sequence_length = len(full_sequence)

    bases_to_predict = sequence_length - min_seed_length

    probability_vector = np.zeros(bases_to_predict)

    remaining_length = bases_to_predict
    model.reset_states()
    current_sequence = str(full_sequence[0:min_seed_length])
    length = min_seed_length

    seed = full_sequence[0:length - 1]
    input_seq = label_encoder.encode_kmers([seed], [], with_shifted_output=False)[0]
    model.predict(input_seq)
    while remaining_length > 0:
        base = current_sequence[length-1:length]
        base_encoding = label_encoder.encode_kmers([base], [], with_shifted_output=False)[0]

        prediction = model.predict(base_encoding)
        random_base_idx = np.random.randint(len(CONSTANTS.ONE_HOT_ENCODING))
        random_base = CONSTANTS.REVERSE_INTEGER_ENCODING[random_base_idx]
        current_sequence += random_base
        probability_vector[length - min_seed_length] = prediction[0][random_base_idx]

        remaining_length -= 1
        length += 1
    return current_sequence, probability_vector