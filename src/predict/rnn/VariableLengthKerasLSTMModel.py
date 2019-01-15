import keras.optimizers as optimizers
from keras.layers import CuDNNLSTM, LSTM, Dense
from keras.models import Sequential

from constants import VariableLengthRnnEncodingConstants as CONSTANTS


class VariableLengthKerasLSTMModel:
    def _initialize_model(self):
        model = Sequential()
        if self.with_gpu:
            #TODO: technically input timesteps is fixed to the maximum length
            model.add(CuDNNLSTM(self.latent_dim, input_shape=(None, self.one_hot_encoding_length)))
        else:
            model.add(LSTM(self.latent_dim, input_shape=(None, self.one_hot_encoding_length)))
        model.add(Dense(self.one_hot_decoding_length, activation='softmax'))

        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])  # TODO: 3 hyperparameters here


    def __init__(self, prediction_length, batch_size=64, epochs=10, latent_dim=100, with_gpu=True):
        self.encoding = CONSTANTS.ONE_HOT_ENCODING
        encoding_length = self.encoding.shape[1]

        self.with_gpu = with_gpu
        self.batch_size = batch_size  # Batch size for training.
        self.epochs = epochs  # Number of epochs to train for.
        self.latent_dim = latent_dim  # Latent dimensionality of the encoding space.
        self.one_hot_encoding_length = encoding_length
        self.one_hot_decoding_length = encoding_length
        self.prediction_length = prediction_length
        self._initialize_model()
        print(self.model.summary())

    def fit(self, X, Y):
        self.model.fit(X, Y,
                  batch_size=self.batch_size,
                  epochs=self.epochs)

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
        y = self.model.predict(X)
        return y