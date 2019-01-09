import keras.optimizers as optimizers
import numpy as np
from keras.layers import CuDNNLSTM, LSTM, Input, Dense
from keras.models import Model

from constants import RnnEncodingConstants as CONSTANTS


#TODO: https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py
#TODO: https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py
#TODO: https://machinelearningmastery.com/define-encoder-decoder-sequence-sequence-model-neural-machine-translation-keras/
class KerasLSTMModel:
    def _encoder_model(self):
        # shape = (None, n) implies that there is a variable number of rows
        encoder_inputs = Input(shape=(None, self.one_hot_encoding_length))
        encoder = CuDNNLSTM(self.latent_dim, return_state=True) if self.with_gpu else LSTM(self.latent_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]
        return encoder, encoder_inputs, encoder_outputs, encoder_states

    def _decoder_model(self, encoder_states):
        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, self.one_hot_decoding_length))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.

        # The decoder begins with the final cell state and hidden state of the encoder
        decoder_lstm = CuDNNLSTM(self.latent_dim, return_sequences=True, return_state=True) if self.with_gpu else LSTM(self.latent_dim, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)
        decoder_dense = Dense(self.one_hot_decoding_length, activation='softmax') #TODO: hyperparameter here
        decoder_outputs = decoder_dense(decoder_outputs)
        return decoder_lstm, decoder_dense, decoder_inputs, decoder_outputs

    def _encoder_inference_model(self, encoder_inputs, encoder_states):
        encoder_model = Model(encoder_inputs, encoder_states)
        return encoder_model

    def _decoder_inference_model(self, decoder_lstm, decoder_inputs, decoder_dense):
        decoder_state_input_h = Input(shape=(self.latent_dim,))
        decoder_state_input_c = Input(shape=(self.latent_dim,))
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_outputs, state_h, state_c = decoder_lstm(
            decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h, state_c]
        decoder_outputs = decoder_dense(decoder_outputs)
        decoder_model = Model(
            [decoder_inputs] + decoder_states_inputs,
            [decoder_outputs] + decoder_states)
        return decoder_model

    def _initialize_models(self):
        # Define an input sequence and process it.
        encoder, encoder_inputs, encoder_outputs, encoder_states = self._encoder_model()

        decoder_lstm, decoder_dense, decoder_inputs, decoder_outputs = self._decoder_model(encoder_states)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
        # plot_model(self.model, to_file='viz/model.png', show_shapes=True)

        # Next: inference mode (sampling).
        # Here's the drill:
        # 1) encode input and retrieve initial decoder state
        # 2) run one step of decoder with this initial state
        # and a "start of sequence" token as target.
        # Output will be the next target token
        # Define sampling models
        self.encoder_inference_model = self._encoder_inference_model(encoder_inputs, encoder_states)
        self.decoder_inference_model = self._decoder_inference_model(decoder_lstm, decoder_inputs, decoder_dense)

        # plot_model(self.encoder_inference_model, to_file='viz/encoder_model.png', show_shapes=True)
        # plot_model(self.decoder_inference_model, to_file='viz/decoder_model.png', show_shapes=True)

        # Run training
        optimizer = optimizers.Adam()
        self.model.compile(optimizer=optimizer,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])  # TODO: 3 hyperparameters here, try learning rate = 0.0001

    def __init__(self, prediction_length, batch_size=64, epochs=10, latent_dim=100, with_gpu=True):
        #TODO: 3 hyperparameters here
        self.encoding = CONSTANTS.ONE_HOT_ENCODING
        encoding_length = self.encoding.shape[1]

        self.with_gpu = with_gpu
        self.batch_size = batch_size  # Batch size for training.
        self.epochs = epochs  # Number of epochs to train for.
        self.latent_dim = latent_dim  # Latent dimensionality of the encoding space.
        self.one_hot_encoding_length = encoding_length
        self.one_hot_decoding_length = encoding_length
        self.prediction_length = prediction_length
        self._initialize_models()
        print(self.model.summary())

    def fit(self, X, Y, shifted_Y):
        self.model.fit([X, shifted_Y], Y,
                  batch_size=self.batch_size,
                  epochs=self.epochs)

    def save_weights(self, path):
        self.model.save_weights(path)

    def load_weights(self, path):
        self.model.load_weights(path)

    def predict(self, X, batch_size=None):
        num_predictions = len(X)

        interval = batch_size if batch_size is not None else num_predictions
        lower_bound = 0
        upper_bound = min(num_predictions, lower_bound + interval)

        decoding = np.zeros((num_predictions, self.prediction_length, self.one_hot_decoding_length))

        while lower_bound < num_predictions:
            batch = X[lower_bound:upper_bound]

            states_to_feed = self.encoder_inference_model.predict(batch)

            # (num_sequences, sequence_length, one_hot_length)
            # Populate the first character of target sequence with the start character.
            start_character_label = CONSTANTS.INTEGER_ENCODING_MAP["!"]
            character_to_feed = CONSTANTS.ONE_HOT_ENCODING[np.full((interval,1), start_character_label)]

            for i in range(self.prediction_length):
                # we initially seed the target seq with the start character and feed in the input sequence
                # to get the model started with a context and a start character
                next_base, state_h, state_c = self.decoder_inference_model.predict([character_to_feed] + states_to_feed)

                decoding[lower_bound:upper_bound, i, :] = next_base[:, 0, :]

                # the input to the next loop are the new states and next base given all the previous loops
                states_to_feed = [state_h, state_c]
                character_to_feed = next_base

            lower_bound = upper_bound
            upper_bound = min(num_predictions, upper_bound + interval)

        return decoding
