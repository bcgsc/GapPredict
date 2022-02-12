import tensorflow.keras as keras
import numpy as np


class TrainingMetric(keras.callbacks.Callback):
    def __init__(self, epochs):
        self.num_epochs = epochs
        self.data = []
        self.batches = []

    def set_generator(self, generator):
        self.generator = generator

    def on_epoch_end(self, epoch, logs=None):
        num_batches = len(self.batches)
        accuracy_per_batch = np.zeros(num_batches)
        batch_sizes = np.zeros(num_batches)
        for i in range(num_batches):
            batch_tuple = self.batches[i]
            input = batch_tuple[0]
            output = batch_tuple[1]
            predictions = self.model.predict(input)
            decoded_output = np.argmax(output, axis=1)
            decoded_predictions = np.argmax(predictions, axis=1)
            matches = np.equal(decoded_output, decoded_predictions)
            accuracy = np.mean(matches)

            accuracy_per_batch[i] = accuracy
            batch_sizes[i] = len(input)

        training_metric = np.sum(accuracy_per_batch*batch_sizes)/np.sum(batch_sizes)
        self.data.append(training_metric)
        self.batches = []

    def on_batch_end(self, batch, logs=None):
        batch = self.generator._pop_earliest_batch()
        self.batches.append(batch)

    def get_data(self):
        return np.array(self.data)