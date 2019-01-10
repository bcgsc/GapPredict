from exceptions.SlidingWindowParamException import SlidingWindowParamException
from preprocess.SlidingWindowExtractor import SlidingWindowExtractor


class VariableLengthKmerExtractor:
    def __init__(self, k_low, spacing, output_length):
        if k_low <= 0 or output_length <= 0:
            raise SlidingWindowParamException("Lengths must be positive")
        if spacing < 0:
            raise SlidingWindowParamException("Spacing cannot be negative")

        self.k_low = k_low
        self.spacing = spacing
        self.output_length = output_length

    def _calculate_upper_bound(self, sequences):
        #TODO: faster to convert to an numpy array then call np.max?
        lengths = list(map(lambda x: len(x), sequences))
        return max(lengths)

    def _init_extractors(self, sequences):
        extractors = []
        k_high = self._calculate_upper_bound(sequences)
        for k in range(self.k_low, k_high + 1):
            extractor = SlidingWindowExtractor(k, self.spacing, self.output_length)
            extractors.append(extractor)

        return extractors, k_high

    def extract_kmers_from_sequence(self, sequences):
        #TODO: may need to use itertools.chain() instead of concatenating all these lists
        inputs = []
        outputs = []
        extractors, k_high = self._init_extractors(sequences)

        for extractor in extractors:
            input_kmers, output_kmers = extractor.extract_kmers_from_sequence(sequences)
            inputs += input_kmers
            outputs += output_kmers

        return inputs, outputs, k_high