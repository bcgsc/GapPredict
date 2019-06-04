import numpy as np

from preprocess.KmerLabelEncoder import KmerLabelEncoder

class BeamSearchPredictor:
    def __init__(self, model):
        self.model = model

    def predict_next_n_bases(self, seed, prediction_length, beam_length):
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
            predictions = self.model.predict(seed_vectors)

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
            tree_breadth = min(tree_breadth, beam_length)

            sorted_idx = np.argsort(next_level_lg_sum_p)[::-1]
            sorted_next_level_seeds = next_level_seeds[sorted_idx]
            sorted_next_level_lg_sum_p = next_level_lg_sum_p[sorted_idx]
            current_level = (sorted_next_level_seeds[:tree_breadth], sorted_next_level_lg_sum_p[:tree_breadth])
        return current_level[0], current_level[1]

    def predict_next_n_bases_greedy(self, seed, prediction_length):
        beam_length = 1
        sequences, probabilities = self.predict_next_n_bases(seed, prediction_length, beam_length)
        return sequences[0], probabilities[0]
