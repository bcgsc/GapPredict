import sys
sys.path.append('../../')

import time

from predict.new_rnn.SingleLSTMModel import SingleLSTMModel
from preprocess.SequenceImporter import SequenceImporter

def main():
    include_reverse_complement = True
    min_seed_length = 26

    arguments = sys.argv[1:]
    paths = arguments if len(arguments) > 0 else ['../data/ecoli_contigs/ecoli_contig_1000.fastq']
    importer = SequenceImporter()
    reads = importer.import_fastq(paths, include_reverse_complement)

    model = SingleLSTMModel(min_seed_length=min_seed_length, stateful=False, batch_size=128, epochs=100, embedding_dim=25, latent_dim=100, with_gpu=True, log_samples=True)

    start_time = time.time()
    model.fit(reads)
    model.save_weights('../weights/my_model_weights.h5')
    end_time = time.time()
    print("Fitting took " + str(end_time - start_time) + "s")

if __name__ == "__main__":
    main()
