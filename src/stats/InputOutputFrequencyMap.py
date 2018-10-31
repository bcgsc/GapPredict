class InputOutputFrequencyMap:
    def __init__(self):
        self.map = {}

    def load_input_outputs(self, inputs, outputs):
        for i in range(len(inputs)):
            input = inputs[i]
            output = outputs[i]
            if input not in self.map:
                self.map[input] = {}
            inner_map = self.map[input]
            if output not in inner_map:
                inner_map[output] = 1
            else:
                inner_map[output] += 1

    def get_unique_mappings_per_input(self):
        mapping_map = {}
        for key in self.map:
            mapping_map[key] = len(self.map[key])
        return mapping_map

    def get_inputs_with_redundant_mappings(self):
        redundants = set()
        for key in self.map:
            if len(self.map[key]) > 1:
                redundants.add(key)
        return redundants

    def get_map(self):
        return self.map
