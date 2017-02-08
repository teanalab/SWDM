from unittest import TestCase

import features.features


class TestIndex(TestCase):
    def setUp(self):
        self.features = features.features.Features('../index/test_files/index')
        self.feature_parameters = {
            "uw_expression_count": {
                "window_size": 17
            },
            "od_expression_count": {
                "window_size": 4
            },
        }

    def test_uw_expression_count(self):
        self.assertEqual(self.features.uw_expression_count("SAMPSON Dog",
                                                           self.feature_parameters["uw_expression_count"]), 2)

    def test_linear_combination(self):
        term = "SAMPSON Dog"
        feature_names = ["uw_expression_count", "od_expression_count"]
        features_weights = {"uw_expression_count": 0.1, "od_expression_count": 0.2}
        res = self.features.linear_combination(term, feature_names, features_weights, self.feature_parameters)
        expected_res = 0.6000000000000001
        self.assertEqual(res, expected_res)
