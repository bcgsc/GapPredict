import numpy as np

class SentenceEmbedder:
    def __init__(self, embedder):
        self.embedder = embedder

    def embed_sentences(self, sentences):
        num_sentences = len(sentences)

        if num_sentences == 0:
            return np.array([])

        num_words = len(sentences[0])
        embedding_length = self.embedder.dimensions

        cube = np.zeros((num_sentences, num_words, embedding_length))

        for i in range(num_sentences):
            for j in range(num_words):
                cube[i][j] = self.embedder.embed(sentences[i][j])

        return cube