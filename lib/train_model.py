import sys
sys.path.append('../../')

import time
import numpy as np
import matplotlib.pyplot as plt
import utils.directory_utils as dir_utils
from lstm.GapPredictModel import GapPredictModel
from preprocess.SequenceImporter import SequenceImporter
from preprocess.SequenceReverser import SequenceReverser

def _plot_training_validation(epochs, accuracy, loss, training_accuracy, training_loss, lengths, directory_path, legend=None, best_epoch=None):
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
    if best_epoch is not None:
        plt.axvline(best_epoch, color='r', linestyle='dashed', linewidth=3)
    fig = plt.savefig(directory_path + 'training_accuracy.png')
    plt.close(fig)

    plt.figure(figsize=figure_dimensions)
    plt.xlim(0, epochs)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.plot(np.arange(len(training_loss)), training_loss, linewidth=3)
    if best_epoch is not None:
        plt.axvline(best_epoch, color='r', linestyle='dashed', linewidth=3)
    fig = plt.savefig(directory_path + 'training_loss.png')
    plt.close(fig)

    plt.figure(figsize=figure_dimensions)
    plt.ylim(0, 1.1)
    plt.xlim(0, epochs)
    plt.xlabel('Epoch')
    plt.ylabel('% Predicted')
    weighted_mean_accuracy = np.sum(accuracy * lengths, axis=1) / np.sum(lengths)
    plt.plot(np.arange(len(weighted_mean_accuracy)), weighted_mean_accuracy, linewidth=3)
    if best_epoch is not None:
        plt.axvline(best_epoch, color='r', linestyle='dashed', linewidth=3)
    if legend is not None:
        plt.legend(legend+["Best Epoch"], loc='best')
    fig = plt.savefig(directory_path + 'validation_accuracy.png')
    plt.close(fig)

    plt.figure(figsize=figure_dimensions)
    plt.xlim(0, epochs)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    weighted_mean_loss = np.sum(loss * lengths, axis=1) / np.sum(lengths)
    plt.plot(np.arange(len(weighted_mean_loss)), weighted_mean_loss, linewidth=3)
    if best_epoch is not None:
        plt.axvline(best_epoch, color='r', linestyle='dashed', linewidth=3)
    if legend is not None:
        plt.legend(legend+["Best Epoch"], loc='best')
    fig = plt.savefig(directory_path + 'validation_loss.png')
    plt.close(fig)

    np.save(directory_path + "training_accuracy", training_accuracy)
    np.save(directory_path + "training_loss", training_loss)
    np.save(directory_path + "validation_accuracy", accuracy)
    np.save(directory_path + "validation_loss", loss)
    np.save(directory_path + "lengths", lengths)

def train_model(base_directory, min_seed_length, reference, reads, epochs, batch_sizes, rnn_dims, embedding_dims, replicates, patience, seed_range_upper, log_samples=False):
    include_reverse_complement = True

    importer = SequenceImporter()
    reverse_complementer = SequenceReverser()
    reads = importer.import_fastq([reads], include_reverse_complement)

    reference_sequence = importer.import_fasta([reference])
    reference_sequences = [reference_sequence[0], reverse_complementer.reverse_complement(reference_sequence[0]),
                           reference_sequence[1], reverse_complementer.reverse_complement(reference_sequence[1])]
    lengths = np.array(list(map(lambda x: len(x), reference_sequences)))

    with_gpu=True
    log_samples=log_samples
    log_training=False
    early_stopping=True
    legend=['Mean Sequence']

    terminal_directory_character = dir_utils.get_terminal_directory_character()

    for batch_size in batch_sizes:
        for embedding_dim in embedding_dims:
            for latent_dim in rnn_dims:
                for i in range(replicates):
                    weights_path = dir_utils.clean_directory_string(base_directory)
                    inner_directory = "BS_" + str(batch_size) + "_ED_" + str(embedding_dim) + "_LD_" + str(latent_dim) \
                                      + "_E_" + str(epochs) + "_R_" + str(i)

                    writer_path = weights_path + inner_directory + terminal_directory_character

                    dir_utils.mkdir(weights_path)
                    dir_utils.mkdir(writer_path)

                    model = GapPredictModel(min_seed_length=min_seed_length, stateful=False,
                                            batch_size=batch_size,
                                            epochs=epochs, embedding_dim=embedding_dim, latent_dim=latent_dim,
                                            with_gpu=with_gpu, log_samples=log_samples,
                                            reference_sequences=reference_sequences,
                                            log_training=log_training, early_stopping=early_stopping,
                                            patience=patience, seed_range_upper=seed_range_upper, base_path=base_directory)

                    start_time = time.time()
                    history = model.fit(reads)
                    model.save_weights(weights_path + 'my_model_weights.h5')
                    end_time = time.time()
                    print("Fitting took " + str(end_time - start_time) + "s")

                    accuracy, loss = model.validation_history()
                    if log_training:
                        training_accuracy = model.training_history()
                    else:
                        training_accuracy = history.history['acc']
                    training_loss = history.history['loss']
                    if early_stopping:
                        best_epoch = model.get_best_epoch()
                    _plot_training_validation(epochs, accuracy, loss, training_accuracy, training_loss, lengths, writer_path, legend=legend, best_epoch=best_epoch)
