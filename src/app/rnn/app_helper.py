import time

import numpy as np

from SequenceMatchCalculator import SequenceMatchCalculator
from onehot.OneHotMatrix import OneHotMatrixDecoder


def get_checkpoints(num_predictions):
    progress_checks = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])*num_predictions
    progress_checks = progress_checks.astype(int)
    return set(progress_checks)

def predict(input, model, bases_to_predict):
    stats = {}
    decoded_predictions = []

    num_predictions = len(input)
    decoder = OneHotMatrixDecoder(bases_to_predict)
    progress_checks = get_checkpoints(num_predictions)

    for i in range(num_predictions):
        if i in progress_checks:
            print(str(int((i*100)/num_predictions)) + "% finished")
        predicted_output = model.predict(input[i:i + 1])
        decoded_predicted_output = decoder.decode_sequences(predicted_output)
        decoded_sequence = ''.join(decoded_predicted_output[0])
        if decoded_sequence not in stats:
            stats[decoded_sequence] = 1
        else:
            stats[decoded_sequence] += 1

        decoded_predictions.append(decoded_predicted_output[0])

    print("Stats = " + str(stats))
    return decoded_predictions

def predict_and_validate(input, output_seq_cube, model, bases_to_predict):
    decoder = OneHotMatrixDecoder(bases_to_predict)
    validator = SequenceMatchCalculator()

    start_time = time.time()
    decoded_actual_output = decoder.decode_sequences(output_seq_cube)

    decoded_predictions = predict(input, model, bases_to_predict)
    end_time = time.time()
    print("Predicting took " + str(end_time - start_time) + "s")

    start_time = time.time()
    matches = validator.compare_sequences(decoded_predictions, decoded_actual_output)

    mean_match = np.mean(matches, axis=0)
    print("Mean Match = " + str(mean_match))

    end_time = time.time()

    print("Validation took " + str(end_time - start_time) + "s")