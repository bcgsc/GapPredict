import numpy as np

import constants.ConstantsHelper as chelper

REVERSE_INTEGER_ENCODING = np.array(["A", "C", "G", "T", "!"])
INTEGER_ENCODING_MAP, BASES, ONE_HOT_ENCODING = chelper.generate_constants_set(REVERSE_INTEGER_ENCODING)
