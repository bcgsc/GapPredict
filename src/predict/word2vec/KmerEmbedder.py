from gensim.models import word2vec, KeyedVectors
import numpy as np

class KmerEmbedder:
    #TODO: what should these hyperparameters be?
    #window = range around each word to use as context
    #min_count = minimum # words before we consider a word
    #size = # dimensions for vector embedding
    def __init__(self, window=10, min_count=5, dimensions=100, workers=4):
        self.model = None
        self.window = window
        self.min_count = min_count
        self.dimensions = dimensions
        self.workers = workers

    def train(self, sentences):
        #TODO: may need to consider making sentences an iterator if this gets too big
        self.model = word2vec.Word2Vec(
            sentences=sentences,
            size=self.dimensions,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            sg=1 #skip-gram
        )

    def embed(self, word):
        if self.model is None:
            return np.array([])
        else:
            return self.model[word]

    def save(self, path, as_text=False):
        if self.model is not None:
            self.model.wv.save_word2vec_format(path + '/model.bin', binary=True)
            if as_text:
                self.model.wv.save_word2vec_format(path + '/model.txt', binary=False)

    def load(self, path, as_binary=True):
        self.model = KeyedVectors.load_word2vec_format(path, binary=as_binary)

    def print_info(self, list_vocab=True):
        if self.model is not None:
            print(self.model)
            if list_vocab:
                print(list(self.model.wv.vocab))
        else:
            print("No model detected")