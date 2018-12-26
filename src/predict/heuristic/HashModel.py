from stats.InputOutputFrequencyMap import InputOutputFrequencyMap

class HashModel:
    def __init__(self):
        self.model = {}

    def fit(self, X, y):
        temp_map = InputOutputFrequencyMap()
        temp_map.load_input_outputs(X, y)
        input_output_map = temp_map.map

        for input_key in input_output_map:
            input_mappings = input_output_map[input_key]
            max_output_key = None
            max_output_count = 0
            for output_key in input_mappings:
                output_count = input_mappings[output_key]
                if output_count > max_output_count:
                    max_output_key = output_key
                    max_output_count = output_count

            self.model[input_key] = max_output_key

    def predict(self, X):
        y = []
        for input in X:
            if input not in self.model or self.model[input] is None:
                y.append("")
            else:
                y.append(self.model[input])
        return y