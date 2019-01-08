import sys
sys.path.append('../../')

import time

from preprocess.SentenceExtractor import SentenceExtractor
from preprocess.SequenceImporter import SequenceImporter
from predict.word2vec.KmerEmbedder import KmerEmbedder

def main():
    importer = SequenceImporter()
    sentence_extractor = SentenceExtractor()

    include_reverse_complement = True
    k = 10

    arguments = sys.argv[1:]
    paths = arguments if len(arguments) > 0 else ['../data/ecoli_contigs/ecoli_contig_1000.fastq']

    start_time = time.time()
    reads = importer.import_fastq(paths, include_reverse_complement)
    end_time = time.time()
    print("Import took " + str(end_time - start_time) + "s")

    start_time = time.time()
    sequences = list(map(lambda parsed_record: parsed_record.sequence, reads))
    end_time = time.time()
    print("Additional preprocessing took " + str(end_time - start_time) + "s")

    start_time = time.time()
    sentences = sentence_extractor.split_sequences_into_kmers(sequences, k)
    end_time = time.time()
    print("Sentence extraction took " + str(end_time - start_time) + "s")

    embedder = KmerEmbedder(window=10, min_count=5, dimensions=100, workers=32)
    start_time = time.time()
    embedder.train(sentences)
    end_time = time.time()
    print("Word embedding took " + str(end_time - start_time) + "s")

    embedder.print_info(list_vocab=False)

    embedder.save('model', as_text=True)

if __name__ == "__main__":
    main()