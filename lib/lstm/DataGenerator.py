import os

import tensorflow.keras.utils
import numpy as np

import utils.directory_utils as UTILS
from constants import EncodingConstants as CONSTANTS
from onehot.OneHotVector import OneHotVectorEncoder
from preprocess.KmerLabelEncoder import KmerLabelEncoder


class DataGenerator(tensorflow.keras.utils.Sequence):
    def __init__(self, reads, min_seed_length, batch_size=64, spacing=0, log_samples=False, log_training=False,
                 seed_range_upper=None, base_path=None):
        self.batch_size = batch_size
        self.min_seed_length = min_seed_length
        self.one_hot_encoder = OneHotVectorEncoder(1, encoding_constants=CONSTANTS)
        self.label_encoder = KmerLabelEncoder()
        self.output_length = 1
        self.log_samples = log_samples
        self.spacing = spacing
        self.log_training = log_training
        self.seed_range_upper=seed_range_upper

        if self.log_training:
            self.batches = []

        file_name = "training.csv"
        if base_path is not None:
            directory_path = UTILS.clean_directory_string(base_path)
        else:
            if os.name == 'nt':
                directory_path = 'E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\lib\\lstm\\new_rnn\\out\\'
            else:
                directory_path = '/home/echen/Desktop/Projects/Sealer_NN/lib/lstm/new_rnn/out/'
        self.output_file_path = directory_path + file_name

        self.reads = np.array(list(filter(lambda x:self._above_minimum_length(x), reads)))

        if self.log_samples:
            self._clean_output_file()

    def _above_minimum_length(self, read):
        minimum_length = self.min_seed_length + self.spacing + self.output_length
        return len(read) >= minimum_length

    def __len__(self):
        max_read_length = max(list(map(lambda x: len(x), self.reads)))
        expected_length = 100
        multiplier = max_read_length/expected_length
        base_iterations = len(self.reads)/self.batch_size
        return int(np.ceil(base_iterations * multiplier))

    def __getitem__(self, index):
        total_reads = len(self.reads)
        idx = np.random.randint(total_reads, size=self.batch_size)

        batch = self.reads[idx]
        X, y = self._process_batch(batch)
        return X, y

    def _clean_output_file(self):
        file = open(self.output_file_path, 'w+')
        file.write('input_bases,output_base\n')
        file.close()

    def _validate_sequence(self, seq):
        is_valid = True
        for l in seq:
            if l not in CONSTANTS.BASES:
                is_valid = False
                break

        return is_valid

    def _save_batch(self, inputs, outputs):
        file = open(self.output_file_path, 'a')
        for i in range(len(inputs)):
            input_string = inputs[i]
            output_string = outputs[i]
            mapping_string = input_string + ',' + output_string + '\n'
            file.write(mapping_string)
        file.close()


    def _process_batch(self, batch):
        min_kmer_length = min(list(map(lambda x:len(x), batch)))
        if self.seed_range_upper is not None and self.seed_range_upper >= self.min_seed_length:
            #can't be longer than the shortest kmer in the batch
            candidate_max_seed_length = self.seed_range_upper + 1
            max_seed_length = min(min_kmer_length, candidate_max_seed_length)
        else:
            max_seed_length = min_kmer_length

        input_length = np.random.randint(low=self.min_seed_length, high=max_seed_length - self.spacing - self.output_length + 1)

        np_inputs, np_outputs = self._extract_random_input_output(batch, input_length)

        if self.log_samples:
            self._save_batch(np_inputs, np_outputs)

        X, y = self.label_encoder.encode_kmers(np_inputs, np_outputs, with_shifted_output=False)[0:2]
        y = self.one_hot_encoder.encode_sequences(y)

        if self.log_training:
            self.batches.append((X, y))

        return X, y

    def _pop_earliest_batch(self):
        batch = self.batches[0]
        self.batches = self.batches[1:]
        return batch

    def _extract_random_input_output(self, batch, input_length):
        inputs = []
        outputs = []

        total_length = input_length + self.spacing + self.output_length
        for i in range(len(batch)):
            read = batch[i]
            read_length = len(read)
            start_idx = np.random.randint(low=0, high=read_length - total_length + 1)
            subread = read[start_idx:start_idx + total_length]
            input_subread = subread[0:input_length]
            output_subread = subread[0 + input_length + self.spacing:total_length]

            if self._validate_sequence(input_subread) and self._validate_sequence(output_subread):
                inputs.append(input_subread)
                outputs.append(output_subread)

        np_inputs = np.array(inputs)
        np_outputs = np.array(outputs)
        return np_inputs, np_outputs


