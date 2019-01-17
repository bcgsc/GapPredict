import keras.utils
import numpy as np

from preprocess.SequenceSplitEncoder import SequenceSplitEncoder


class DataGenerator(keras.utils.Sequence):
    def __init__(self, reads, min_seed_length, batch_size=64):
        self.reads = reads
        self.batch_size = batch_size
        self.min_seed_length = min_seed_length

    def __len__(self):
        return int(np.ceil(len(self.reads)/self.batch_size))

    def __getitem__(self, index):
        batch_idx = np.random.randint(len(self.reads), size=self.batch_size)
        batch = self.reads[batch_idx]
        X, y, shifted_y = self._process_batch(batch)
        return [X, shifted_y], y

    def _process_batch(self, batch):
        min_kmer_length = min(list(map(lambda x:len(x), batch)))
        split_index = np.random.randint(low=self.min_seed_length, high=min_kmer_length)

        splitter = SequenceSplitEncoder(split_index)
        X, y, shifted_y = splitter.split_sequences(batch)
        return X, y, shifted_y


