import time

import numpy as np

from preprocess.SequenceMatchCalculator import SequenceMatchCalculator
from onehot.OneHotMatrix import OneHotMatrixDecoder

def predict(X, model, bases_to_predict):
    decoder = OneHotMatrixDecoder(bases_to_predict)

    predicted_output = model.predict(X)

    decoded_predictions = decoder.decode_sequences(predicted_output)
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