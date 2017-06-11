from unittest import TestCase

import features.features
from parameters.parameters import Parameters

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 11 / 21 / 16


class TestFeatures(TestCase):
    def setUp(self):
        self.parameters = Parameters()
        self.parameters.params["repo_dir"] = '../index/test_files/index'

        self.features = features.features.Features(self.parameters)
        self.feature_parameters = {
            "uw_expression_norm_count": {
                "window_size": 17
            },
            "uw_expression_norm_document_count": {
                "window_size": 4
            }
        }

    def test_linear_combination(self):
        gram_pair = (('two', 'stand'), (('pair', 0.5850129127502441), ('sit', 0.5178298950195312)))
        feature_names = ["uw_expression_norm_count", "uw_expression_norm_document_count"]
        features_weights = {"uw_expression_norm_count": 0.1, "uw_expression_norm_document_count": 0.2}
        res = self.features.linear_combination(gram_pair, feature_names, features_weights, self.feature_parameters)
        expected_res = 0.8793005091297531
        self.assertEqual(res, expected_res)
