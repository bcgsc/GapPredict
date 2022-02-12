import tensorflow.keras.optimizers as optimizers
#from keras.utils import plot_model
from tensorflow.keras.layers import LSTM, Embedding, Dense
from tensorflow.keras.models import Sequential

from constants import EncodingConstants as CONSTANTS
from lstm.DataGenerator import DataGenerator
from lstm.TrainingMetric import TrainingMetric


class GapPredictModel:
    def _initialize_models(self):
        model = Sequential()
        if self.stateful:
            batch_input_shape = (self.batch_size, None)
            model.add(Embedding(batch_input_shape=batch_input_shape, input_dim=self.one_hot_encoding_length,
                                output_dim=self.embedding_dim, input_length=None))
        else:
            model.add(Embedding(input_dim=self.one_hot_encoding_length,
                                output_dim=self.embedding_dim, input_length=None))

        model.add(LSTM(self.latent_dim, stateful=self.stateful, input_shape=(None, self.embedding_dim)))
        model.add(Dense(self.one_hot_decoding_length, activation='softmax'))
        self.model = model

        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def __init__(self, min_seed_length, spacing=0, batch_size=64, stateful=False, epochs=100, embedding_dim=25,
                 latent_dim=100, with_gpu=True, log_samples=True, reference_sequences=None, log_training=False,
                 early_stopping=False, patience=200, seed_range_upper=None, base_path=None):
        self.encoding = CONSTANTS.ONE_HOT_ENCODING
        encoding_length = self.encoding.shape[1]

        self.stateful = stateful
        self.with_gpu = with_gpu # deprecated
        self.batch_size = batch_size  # Batch size for training.
        self.epochs = epochs  # Number of epochs to train for.
        self.latent_dim = latent_dim  # Latent dimensionality of the encoding space.
        self.embedding_dim = embedding_dim
        self.one_hot_encoding_length = encoding_length
        self.one_hot_decoding_length = encoding_length
        self.min_seed_length = min_seed_length
        self.log_samples = log_samples
        self.spacing = spacing
        self.early_stopping = early_stopping
        self.patience = patience
        self.seed_range_upper = seed_range_upper
        self.base_path = base_path
        self._initialize_models()

        #plot_model(self.model, to_file="model.png", show_shapes=True)

        self.log_training = log_training
        self.callbacks = []
        if reference_sequences is not None:
            from lstm.ValidationMetric import ValidationMetric
            self.validator = ValidationMetric(reference_sequences, self.min_seed_length, self.spacing,
                                              self.embedding_dim, self.latent_dim, self.epochs, self.patience,
                                                early_stopping=self.early_stopping)
            self.callbacks.append(self.validator)
        if self.log_training:
            self.train_validator = TrainingMetric(self.epochs)
            self.callbacks.append(self.train_validator)

        print(self.model.summary())

    def fit(self, X):
        generator = DataGenerator(X, self.min_seed_length, self.batch_size, log_samples=self.log_samples,
                                  spacing=self.spacing, log_training=self.log_training, seed_range_upper=self.seed_range_upper,
                                  base_path=self.base_path)
        if self.log_training:
            self.train_validator.set_generator(generator)
        history = self.model.fit_generator(generator, epochs=self.epochs, callbacks=self.callbacks)
        return history

    def reset_states(self):
        if self.model is not None:
            self.model.reset_states()

    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path):
        self.model.load_weights(path)

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def get_weights(self):
        return self.model.get_weights()

    def predict(self, X):
        return self.model.predict(X, batch_size=len(X))

    def validation_history(self):
        return self.validator.get_data()

    def get_best_epoch(self):
        return self.validator.get_best_epoch()

    def training_history(self):
        return self.train_validator.get_data()