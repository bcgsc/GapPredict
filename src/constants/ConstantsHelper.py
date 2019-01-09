import numpy as np

def generate_constants_set(reverse_integer_encoding):
    integer_encoding_map = {}

    for i in range(len(reverse_integer_encoding)):
        integer_encoding_map[reverse_integer_encoding[i]] = i

    bases = {}
    for base in ["A", "C", "G", "T"]:
        bases[base] = integer_encoding_map[base]

    one_hot_encoding = np.zeros((len(reverse_integer_encoding), len(reverse_integer_encoding)))
    for i in range(len(one_hot_encoding)):
        one_hot_encoding[i][i] = 1

    return integer_encoding_map, bases, one_hot_encoding