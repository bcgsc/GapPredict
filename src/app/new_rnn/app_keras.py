import sys
sys.path.append('../../')

import time
import numpy as np
import matplotlib.pyplot as plt

from predict.new_rnn.SingleLSTMModel import SingleLSTMModel
from preprocess.SequenceImporter import SequenceImporter

def _plot_training_validation(epochs, validation_metrics, training_accuracy):
    root_path = 'E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\src\\app\\new_rnn\\out\\'
    #root_path = '/home/echen/Desktop/Projects/Sealer_NN/src/app/new_rnn/out/'

    plt.figure()
    plt.ylim(0, 1.1)
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.plot(epochs, training_accuracy)
    plt.savefig(root_path + 'training_accuracy.png')
    plt.clf()

    plt.figure()
    plt.ylim(0, 1.1)
    plt.xlabel('Epoch')
    plt.ylabel('Fraction Until Mismatch')
    plt.plot(epochs, validation_metrics)
    plt.savefig(root_path + 'percent_predicted_till_mismatch.png')
    plt.clf()

def main():
    include_reverse_complement = True
    min_seed_length = 26
    spacing = 10

    arguments = sys.argv[1:]
    paths = arguments if len(arguments) > 0 else ['../data/ecoli_contigs/ecoli_contig_1000.fastq']
    importer = SequenceImporter()
    reads = importer.import_fastq(paths, include_reverse_complement)

    path = '../data/ecoli_contigs/ecoli_contig_1000.fasta'
    reference_sequence = importer.import_fasta([path])[0]

    model = SingleLSTMModel(min_seed_length=min_seed_length, spacing=spacing, stateful=False, batch_size=128, epochs=100, embedding_dim=25, latent_dim=100, with_gpu=True, log_samples=True, reference_sequence=reference_sequence)

    start_time = time.time()
    history = model.fit(reads)
    model.save_weights('../weights/my_model_weights.h5')
    end_time = time.time()
    print("Fitting took " + str(end_time - start_time) + "s")

    epochs = np.array(history.epoch)
    validation_metrics = model.validation_history()[0]
    training_accuracy = np.array(history.history['acc'])
    _plot_training_validation(epochs, validation_metrics, training_accuracy)

if __name__ == "__main__":
    main()
