from sklearn.preprocessing import OneHotEncoder

class NonpositiveLengthException(Exception):
    pass

class OneHotVectorEncoder:
    def __init__(self, input_length):
        if(input_length < 1):
            raise NonpositiveLengthException
        self.encoder = OneHotEncoder(sparse=False)
        self.input_length = input_length
        self._train_encoder()

    #TODO: this does not support quality, to be honest we probably don't even want this file
    def _train_encoder(self):
        row_a = []
        row_c = []
        row_g = []
        row_t = []
        for i in range(self.input_length):
            row_a.append("A")
            row_c.append("C")
            row_g.append("G")
            row_t.append("T")
        encoder_train_matrix = [row_a, row_c, row_g, row_t]
        self.encoder.fit(encoder_train_matrix)

    def encode_sequences(self, sequences):
        return self.encoder.transform(sequences)

    def decode_sequences(self, encoded_sequences):
        return self.encoder.inverse_transform(encoded_sequences)