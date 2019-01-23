import keras.optimizers as optimizers
import numpy as np
from keras.layers import CuDNNLSTM, LSTM, Embedding, Dense
from keras.models import Sequential

from constants import EncodingConstants as CONSTANTS
from predict.new_rnn.DataGenerator import DataGenerator

class SingleLSTMModel:
    def _initialize_models(self):
        model = Sequential()
        if self.stateful:
            batch_input_shape = (self.batch_size, None)
            model.add(Embedding(batch_input_shape=batch_input_shape, input_dim=self.one_hot_encoding_length,
                                output_dim=self.embedding_dim, input_length=None))
        else:
            model.add(Embedding(input_dim=self.one_hot_encoding_length,
                                output_dim=self.embedding_dim, input_length=None))
        if self.with_gpu:
            model.add(CuDNNLSTM(self.latent_dim, stateful=self.stateful, input_shape=(None, self.embedding_dim)))
        else:
            model.add(LSTM(self.latent_dim, stateful=self.stateful, input_shape=(None, self.embedding_dim)))
        model.add(Dense(self.one_hot_decoding_length, activation='softmax'))
        self.model = model

        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def __init__(self, min_seed_length, batch_size=64, stateful=False, epochs=100, embedding_dim=25, latent_dim=100, with_gpu=True):
        self.encoding = CONSTANTS.ONE_HOT_ENCODING
        encoding_length = self.encoding.shape[1]

        self.stateful = stateful
        self.with_gpu = with_gpu
        self.batch_size = batch_size  # Batch size for training.
        self.epochs = epochs  # Number of epochs to train for.
        self.latent_dim = latent_dim  # Latent dimensionality of the encoding space.
        self.embedding_dim = embedding_dim
        self.one_hot_encoding_length = encoding_length
        self.one_hot_decoding_length = encoding_length
        self.min_seed_length = min_seed_length
        self._initialize_models()

        print(self.model.summary())

    def fit(self, X):
        generator = DataGenerator(X, self.min_seed_length, self.batch_size)
        self.model.fit_generator(generator, epochs=self.epochs)

    def save_weights(self, path):
        if self.model is not None:
            self.model.save_weights(path)
        else:
            print("No model")

    def load_weights(self, path):
        if self.model is not None:
            self.model.load_weights(path)
        else:
            print("No model")

    def predict(self, X):
        return self.model.predict(X, batch_size=len(X))
