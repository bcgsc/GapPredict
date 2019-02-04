import keras.optimizers as optimizers
from keras.layers import CuDNNLSTM, LSTM, Embedding, Dense
from keras.models import Sequential

from constants import EncodingConstants as CONSTANTS
from predict.new_rnn.DataGenerator import DataGenerator
from predict.new_rnn.ValidationMetric import ValidationMetric


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

    def __init__(self, min_seed_length, spacing=0, batch_size=64, stateful=False, epochs=100, embedding_dim=25, latent_dim=100, with_gpu=True, log_samples=True, reference_sequence=None):
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
        self.log_samples = log_samples
        self.spacing = spacing
        self._initialize_models()
        if reference_sequence is not None:
            self.validator = ValidationMetric(self.model, reference_sequence, self.min_seed_length, self.spacing)
            self.callbacks = [self.validator]

        print(self.model.summary())

    def fit(self, X):
        generator = DataGenerator(X, self.min_seed_length, self.batch_size, log_samples=self.log_samples, spacing=self.spacing)
        history = self.model.fit_generator(generator, epochs=self.epochs, callbacks=self.callbacks)
        return history

    def reset_states(self):
        if self.model is not None:
            self.model.reset_states()

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

    def validation_history(self):
        return self.validator.get_data()
