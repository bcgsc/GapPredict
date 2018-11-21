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

    def get_total_unique_mappings_per_input(self):
        mapping_map = self.get_unique_mappings_per_input()
        sum = 0
        for key in mapping_map:
            sum += mapping_map[key]
        return sum

    def get_inputs_with_redundant_mappings(self):
        redundants = set()
        for key in self.map:
            if len(self.map[key]) > 1:
                redundants.add(key)
        return redundants

    def _sum_map(self, map):
        acc = 0
        for key in map:
            acc += map[key]
        return acc

    def get_input_stats(self):
        input_stats = {}
        for key in self.map:
            input_stats[key] = self._sum_map(self.map[key])
        return input_stats

    def get_output_stats(self):
        output_stats = {}
        for key in self.map:
            inner_map = self.map[key]
            for inner_key in inner_map:
                if inner_key not in output_stats:
                    output_stats[inner_key] = inner_map[inner_key]
                else:
                    output_stats[inner_key] += inner_map[inner_key]
        return output_stats

    def get_map(self):
        return self.map
