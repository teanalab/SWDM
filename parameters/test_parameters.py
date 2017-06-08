import sys
from unittest import TestCase

import numpy as np

from parameters.parameters import Parameters

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 11 / 21 / 16


class TestParameters(TestCase):
    def setUp(self):
        self.parameters = Parameters()
        self.parameters.params_file = "test_files/parameters.json"

    def test_write_to_parameters_file(self):
        params_ = {
            "norm_term_count": {"abc": float(np.int64(0.2))
                                },
            "norm_document_count": {"edc": 12
                                    },
            "unigrams_cosine_similarity_with_orig": {"def": 1.2
                                                     },
        }
        self.parameters.params = params_
        self.parameters.write_to_parameters_file(self.parameters.params_file)
        self.parameters.read_from_params_file()
        print(type(self.parameters.params["norm_term_count"]["abc"]), file=sys.stderr)
        self.assertEqual(params_, self.parameters.params)
