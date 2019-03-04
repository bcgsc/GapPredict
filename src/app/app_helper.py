import time

from preprocess.SequenceImporter import SequenceImporter
from preprocess.SlidingWindowExtractor import SlidingWindowExtractor


def extract_kmers(paths, input_length, spacing, bases_to_predict, include_reverse_complement, unique):
    importer = SequenceImporter()
    extractor = SlidingWindowExtractor(input_length, spacing, bases_to_predict)

    start_time = time.time()
    reads = importer.import_fastq(paths, include_reverse_complement)
    end_time = time.time()
    print("Import took " + str(end_time - start_time) + "s")

    start_time = time.time()
    input_kmers, output_kmers = extractor.extract_kmers_from_sequence(reads, unique=unique)
    end_time = time.time()
    print("Extraction took " + str(end_time - start_time) + "s")
    return input_kmers, output_kmers