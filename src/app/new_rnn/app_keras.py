import sys
sys.path.append('../../')

import time
import numpy as np
import matplotlib.pyplot as plt
import os

if os.name != 'nt':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
    #change this if someone is competing for GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "1";

from predict.new_rnn.SingleLSTMModel import SingleLSTMModel
from preprocess.SequenceImporter import SequenceImporter
from preprocess.SequenceReverser import SequenceReverser

def _plot_training_validation(epochs, validation_metrics, training_accuracy, directory_name, lengths, legend=None, best_epoch=None):
    if os.name == 'nt':
        root_path = 'E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\src\\app\\new_rnn\\out\\training_metrics\\'
        root_path += directory_name + '\\'
    else:
        root_path = '/home/echen/Desktop/Projects/Sealer_NN/src/app/new_rnn/out/training_metrics/'
        root_path += directory_name + "/"

    if not os.path.exists(root_path):
        os.makedirs(root_path)

    np.save(root_path + "training", training_accuracy)
    np.save(root_path + "validation", validation_metrics)

    plt.rc('xtick', labelsize=28)
    plt.rc('ytick', labelsize=28)
    font = {
        'size': 30
    }
    plt.rc('font', **font)

    figure_dimensions=(18, 12)

    plt.figure(figsize=figure_dimensions)
    plt.ylim(0, 1.1)
    plt.xlim(0, epochs)
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy')
    plt.plot(np.arange(len(training_accuracy)), training_accuracy, linewidth=3)
    plt.savefig(root_path + 'training_accuracy.png')
    plt.clf()

    plt.figure(figsize=figure_dimensions)
    plt.ylim(0, 1.1)
    plt.xlim(0, epochs)
    plt.xlabel('Epoch')
    plt.ylabel('% Predicted')
    mean = np.sum(validation_metrics * lengths, axis=1) / np.sum(lengths)
    plt.plot(np.arange(len(mean)), mean, linewidth=3)
    if best_epoch is not None:
        plt.axvline(best_epoch, color='r', linestyle='dashed', linewidth=3)
    if legend is not None:
        plt.legend(legend+["Best Epoch"], loc='best')
    plt.savefig(root_path + 'percent_predicted_till_mismatch.png')
    plt.clf()

def main():
    include_reverse_complement = True
    min_seed_length = 26
    spacing = 0

    arguments = sys.argv[1:]
    paths = arguments if len(arguments) > 0 else ['../data/real_gaps/sealer_filled/7465348_13506-14596.fastq']
    importer = SequenceImporter()
    reverse_complementer = SequenceReverser()
    reads = importer.import_fastq(paths, include_reverse_complement)

    path = '../data/real_gaps/sealer_filled/7465348_13506-14596.fasta'
    reference_sequence = importer.import_fasta([path])
    reference_sequences = [reference_sequence[0], reverse_complementer.reverse_complement(reference_sequence[0]),
                           reference_sequence[1], reverse_complementer.reverse_complement(reference_sequence[1])]
    lengths = np.array(list(map(lambda x: len(x), reference_sequences)))

    with_gpu=True
    log_samples=False
    log_training=False
    epochs = 1000
    replicates = 2
    early_stopping=True
    legend=['Mean Sequence']

    # (128, 1024, 1024) probably don't go further than this
    # doubling latent_dim seems to increase # parameters by ~3X
    # doubling embedding_dim seems to increase # parameters by ~1.5X
    batch_sizes = [128]
    rnn_dims = [512]
    embedding_dims = [128]

    for batch_size in batch_sizes:
        for embedding_dim in embedding_dims:
            for latent_dim in rnn_dims:
                for i in range(replicates):
                    directory_name = "BS_" + str(batch_size) + "_ED_" + str(embedding_dim) + "_LD_" + str(latent_dim) \
                                     + "_E_" + str(epochs) + "_R_" + str(i)
                    model = SingleLSTMModel(min_seed_length=min_seed_length, spacing=spacing, stateful=False,
                                            batch_size=batch_size,
                                            epochs=epochs, embedding_dim=embedding_dim, latent_dim=latent_dim,
                                            with_gpu=with_gpu, log_samples=log_samples,
                                            reference_sequences=reference_sequences,
                                            log_training=log_training, early_stopping=early_stopping)

                    start_time = time.time()
                    history = model.fit(reads)
                    model.save_weights('../weights/my_model_weights.h5')
                    end_time = time.time()
                    print("Fitting took " + str(end_time - start_time) + "s")

                    validation_metrics = model.validation_history()
                    if log_training:
                        training_accuracy = model.training_history()
                    else:
                        training_accuracy = history.history['acc']
                    if early_stopping:
                        best_epoch = model.get_best_epoch()
                    _plot_training_validation(epochs, validation_metrics, training_accuracy, directory_name, lengths, legend=legend, best_epoch=best_epoch)



if __name__ == "__main__":
    main()
