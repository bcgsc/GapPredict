import keras.utils
import numpy as np

from preprocess.SequenceSplitEncoder import SequenceSplitEncoder
from onehot.OneHotVector import OneHotVectorEncoder
from constants import EncodingConstants as CONSTANTS

class DataGenerator(keras.utils.Sequence):
    def __init__(self, reads, min_seed_length, batch_size=64):
        self.reads = reads
        self.batch_size = batch_size
        self.min_seed_length = min_seed_length
        self.encoder = OneHotVectorEncoder(1, encoding_constants=CONSTANTS)
        self.output_length = 1

    def __len__(self):
        return int(np.ceil(len(self.reads)/self.batch_size))

    def __getitem__(self, index):
        batch_idx = np.random.randint(len(self.reads), size=self.batch_size)
        batch = self.reads[batch_idx]
        X, y = self._process_batch(batch)
        return X, y

    def _process_batch(self, batch):
        min_kmer_length = min(list(map(lambda x:len(x), batch)))
        input_length = np.random.randint(low=self.min_seed_length, high=min_kmer_length - self.output_length + 1)
        total_length = input_length + self.output_length

        np_processed_batch = self._extract_random_input_output(batch, total_length)

        splitter = SequenceSplitEncoder(input_length)
        X, y = splitter.split_sequences(np_processed_batch, output_length=self.output_length)
        y = self.encoder.encode_sequences(y)
        return X, y

    def _extract_random_input_output(self, batch, total_length):
        processed_batch = []

        for i in range(len(batch)):
            read = batch[i]
            read_length = len(read)
            start_idx = np.random.randint(low=0, high=read_length - total_length + 1)
            sub_read = read[start_idx:start_idx + total_length]
            processed_batch.append(sub_read)

        np_processed_batch = np.array(processed_batch)
        return np_processed_batch


