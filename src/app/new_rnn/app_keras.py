import sys
sys.path.append('../../')

import time
import numpy as np
import matplotlib.pyplot as plt
import os

from predict.new_rnn.SingleLSTMModel import SingleLSTMModel
from preprocess.SequenceImporter import SequenceImporter

def _plot_training_validation(epochs, validation_metrics, training_accuracy, directory_name):
    if os.name == 'nt':
        root_path = 'E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\src\\app\\new_rnn\\out\\training_metrics\\'
        root_path += directory_name + '\\'
    else:
        root_path = '/home/echen/Desktop/Projects/Sealer_NN/src/app/new_rnn/out/training_metrics/'
        root_path += directory_name + "/"

    if not os.path.exists(root_path):
        os.makedirs(root_path)

    plt.figure(figsize=(24, 18))
    plt.ylim(0, 1.1)
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.plot(epochs, training_accuracy)
    plt.savefig(root_path + 'training_accuracy.png')
    plt.clf()

    plt.figure(figsize=(24, 18))
    plt.ylim(0, 1.1)
    plt.xlabel('Epoch')
    plt.ylabel('Fraction Until Mismatch')
    plt.plot(epochs, validation_metrics)
    plt.savefig(root_path + 'percent_predicted_till_mismatch.png')
    plt.clf()

def main():
    include_reverse_complement = True
    min_seed_length = 26
    spacing = 0

    arguments = sys.argv[1:]
    paths = arguments if len(arguments) > 0 else ['../data/ecoli_contigs/ecoli_contig_1000.fastq']
    importer = SequenceImporter()
    reads = importer.import_fastq(paths, include_reverse_complement)

    path = '../data/ecoli_contigs/ecoli_contig_1000.fasta'
    reference_sequence = importer.import_fasta([path])[0]

    with_gpu=True
    log_samples=False
    epochs = 1000
    replicates = 2

    # (128, 1024, 1024) probably don't go further than this
    # doubling latent_dim seems to increase # parameters by ~3X
    # doubling embedding_dim seems to increase # parameters by ~1.5X
    batch_sizes = [64, 128, 256]
    embedding_dims = [32, 128, 512]
    latent_dims = [32, 128, 512]
    for batch_size in batch_sizes:
        for embedding_dim in embedding_dims:
            for latent_dim in latent_dims:
                for i in range(replicates):
                    model = SingleLSTMModel(min_seed_length=min_seed_length, spacing=spacing, stateful=False,
                                            batch_size=batch_size,
                                            epochs=epochs, embedding_dim=embedding_dim, latent_dim=latent_dim,
                                            with_gpu=with_gpu,
                                            log_samples=log_samples, reference_sequence=reference_sequence)

                    start_time = time.time()
                    history = model.fit(reads)
                    model.save_weights('../weights/my_model_weights.h5')
                    end_time = time.time()
                    print("Fitting took " + str(end_time - start_time) + "s")

                    epoch_axis = np.array(history.epoch)
                    validation_metrics = model.validation_history()[0]
                    training_accuracy = np.array(history.history['acc'])
                    directory_name = "BS_" + str(batch_size) + "_ED_" + str(embedding_dim) + "_LD_" + str(latent_dim) \
                                     + "_E_" + str(epochs) + "_R_" + str(i)
                    _plot_training_validation(epoch_axis, validation_metrics, training_accuracy, directory_name)



if __name__ == "__main__":
    main()
