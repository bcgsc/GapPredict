import keras.utils
import numpy as np

from constants import EncodingConstants as CONSTANTS
from onehot.OneHotVector import OneHotVectorEncoder
from preprocess.KmerLabelEncoder import KmerLabelEncoder

output_file = 'E:\\Users\\Documents\\School Year 18-19\\Term 1\\CPSC 449\\Sealer_NN\\src\\predict\\new_rnn\\out\\training.csv'
#output_file = '/home/echen/Desktop/Projects/Sealer_NN/src/predict/new_rnn/out/training.csv'

class DataGenerator(keras.utils.Sequence):
    def __init__(self, reads, min_seed_length, batch_size=64, spacing=0, log_samples=False):
        self.batch_size = batch_size
        self.min_seed_length = min_seed_length
        self.encoder = OneHotVectorEncoder(1, encoding_constants=CONSTANTS)
        self.label_encoder = KmerLabelEncoder()
        self.output_length = 1
        self.log_samples = log_samples
        self.spacing = spacing
        if self.log_samples:
            self._clean_output_file()
        self.reads = np.array(list(filter(lambda x:self._above_minimum_length(x), reads)))

    def _above_minimum_length(self, read):
        minimum_length = self.min_seed_length + self.spacing + self.output_length
        return len(read) >= minimum_length


    def __len__(self):
        return int(np.ceil(len(self.reads)/self.batch_size))

    def __getitem__(self, index):
        total_reads = len(self.reads)
        idx = np.arange(total_reads)
        np.random.shuffle(idx)

        batch_idx = idx[0:self.batch_size]
        batch = self.reads[batch_idx]
        X, y = self._process_batch(batch)
        return X, y

    def _clean_output_file(self):
        file = open(output_file, 'w+')
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
        file = open(output_file, 'a')
        for i in range(len(inputs)):
            input_string = inputs[i]
            output_string = outputs[i]
            mapping_string = input_string + ',' + output_string + '\n'
            file.write(mapping_string)
        file.close()


    def _process_batch(self, batch):
        min_kmer_length = min(list(map(lambda x:len(x), batch)))
        input_length = np.random.randint(low=self.min_seed_length, high=min_kmer_length - self.spacing - self.output_length + 1)

        np_inputs, np_outputs = self._extract_random_input_output(batch, input_length)

        if self.log_samples:
            self._save_batch(np_inputs, np_outputs)

        X, y = self.label_encoder.encode_kmers(np_inputs, np_outputs, with_shifted_output=False)[0:2]
        y = self.encoder.encode_sequences(y)
        return X, y

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


