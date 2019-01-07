import sys
sys.path.append('../../')

from predict.word2vec.KmerEmbedder import KmerEmbedder

def main():
    embedder = KmerEmbedder(window=10, min_count=5, dimensions=100)
    embedder.load('model/model.bin', as_binary=True)

    embedder.print_info(list_vocab=True)

if __name__ == "__main__":
    main()