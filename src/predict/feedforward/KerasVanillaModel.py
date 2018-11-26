import keras.optimizers as optimizers
from keras.layers import Dense
from keras.models import Sequential

from constants import EncodingConstants as CONSTANTS


#TODO: https://www.pyimagesearch.com/2016/09/26/a-simple-neural-network-with-python-and-keras/

class KerasVanillaModel:
    def _deep_vanilla_network(self):
        model = Sequential()
        #TODO: consider dropout
        model.add(Dense(200, input_dim=self.input_length, activation="relu", bias_initializer="zeros"))
        model.add(Dense(100, activation="relu", bias_initializer="zeros"))
        model.add(Dense(50, activation="relu", bias_initializer="zeros"))
        model.add(Dense(self.prediction_length, activation="softmax", bias_initializer="zeros"))
        return model

    def _initialize_models(self):
        self.model = self._deep_vanilla_network()
        #plot_model(self.model, to_file='viz/dnn_model.png', show_shapes=True)

        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy']) # TODO: 3 hyperparameters here, try learning rate = 0.0001

    def __init__(self, input_length, prediction_length, batch_size=64, epochs=10):
        # TODO: 2 hyperparameters here, try learning rate = 0.0001
        encoding_length = CONSTANTS.ONE_HOT_ENCODING.shape[1]

        self.batch_size = batch_size
        self.epochs = epochs
        self.input_length = input_length * encoding_length
        self.prediction_length = prediction_length * encoding_length

        self._initialize_models()

    def fit(self, X, y):
        self.model.fit(X, y,
                  batch_size=self.batch_size,
                  epochs=self.epochs)

    def predict(self, X):
        predictions = self.model.predict(X, batch_size=64) #TODO: might be able to tweak this for speed
        return predictions

    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path):
        self.model.load_weights(path)