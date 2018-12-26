import numpy as np
from unittest import TestCase

from preprocess.BaseQualityConverter import BaseQualityConverter


class TestBaseQualityConverter(TestCase):
    def setUp(self):
        self.converter = BaseQualityConverter()

    def test_convert_quality_to_phred(self):
        phred_quality = self.converter.convert_quality_to_phred("===+=AA=AD")
        np.testing.assert_array_equal(phred_quality, np.array([28, 28, 28, 10, 28, 32, 32, 28, 32, 35]))
