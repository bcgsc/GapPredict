class ParsedFastaRecord:
    def __init__(self, id, sequence):
        self.id = id
        self.sequence = sequence

    def __eq__(self, other):
        return self.id == other.id and self.sequence == other.sequence