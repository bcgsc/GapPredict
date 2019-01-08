from unittest import TestCase

import numpy as np

from predict.word2vec.KmerEmbedder import KmerEmbedder
from preprocess.SentenceEmbedder import SentenceEmbedder


class TestSentenceEmbedder(TestCase):
    def setUp(self):
        embedder = KmerEmbedder()
        embedder.load('word2vec_model/model.txt', as_binary=False)
        self.sentence_embedder = SentenceEmbedder(embedder)

    def test_embed_nothing(self):
        sentences = []
        embedding = self.sentence_embedder.embed_sentences(sentences)
        expected_embedding = np.array([])
        np.testing.assert_array_equal(expected_embedding, embedding)

    def test_embed_empty_sentences(self):
        sentences = [
            [],
            []
        ]
        embedding = self.sentence_embedder.embed_sentences(sentences)
        expected_embedding = np.zeros((2, 0, 10))
        np.testing.assert_array_equal(expected_embedding, embedding)

    def test_embed_sentences(self):
        sentences = [
            ["AAAACCCCAG", "GTGGCGATGG", "GCCATCGCCA"],
            ["CATATTCCCT", "GGCTTATCTC", "TGGACCTACT"]
        ]
        embedding = self.sentence_embedder.embed_sentences(sentences)
        expected_embedding = np.array([
            [
                [1.0040596, -1.9837669, 0.46591672, -1.6059558, 2.4772995, 0.89813906, 0.8046972, 2.3206499, -0.45737416, 2.19043],
                [2.6138217, 0.47038797, 0.22518177, 0.7154913, 1.6973671, -3.7398272, 1.5416981, 0.63659227, -0.42228785, -0.3055083],
                [0.7247483, 1.5705863, -2.398422, -0.121020146, 1.9917789, -1.4141414, -0.18104865, 0.4300784, 1.0624833, 3.5454726]
            ],
            [
                [-1.1100093, 0.3674966, 1.0379875, -0.10841686, 3.1394284, -0.28391635, 2.9618638, 0.63555664, -0.3089041, 2.232381],
                [-0.53667307, 1.2379266, 1.9024704, -0.7285193, 3.0646203, -1.6315062, -1.5276833, -1.5291907, 0.46777594, 1.7316724],
                [-0.5859429, 2.4861524, 0.023088513, 0.066931546, 3.8188987, 0.062695764, -0.18682416, -0.43606716, 0.42026728, 2.2893202]
            ]
        ])
        np.testing.assert_array_almost_equal(expected_embedding, embedding, 7)
