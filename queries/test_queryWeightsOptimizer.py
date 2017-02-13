from unittest import TestCase

import sys
from mock import MagicMock

from parameters.parameters import Parameters
from queries.queryWeightsOptimizer import QueryWeightsOptimizer


class TestQueryWeightsOptimizer(TestCase):
    def setUp(self):
        self.parameters = Parameters()

        self.query_weights_optimizer = QueryWeightsOptimizer(self.parameters)

    def test_update_nested_map(self):
        d = {'m': {'d': {'v': {'w': 1}}}}
        res = self.query_weights_optimizer.update_nested_dict(d, 10, ['m', 'd', 'v', 'w'])
        expected_res = {'m': {'d': {'v': {'w': 10}}}}
        self.assertEqual(res, expected_res)

        res = self.query_weights_optimizer.update_nested_dict(d, 10, ['m', 'd', 'v', 'z'])
        expected_res = {'m': {'d': {'v': {'w': 1, 'z': 10}}}}
        self.assertEqual(res, expected_res)

        res = self.query_weights_optimizer.update_nested_dict(d, 10, ['m', 'd', 'l'])
        expected_res = {'m': {'d': {'v': {'w': 1}, 'l': 10}}}
        self.assertEqual(res, expected_res)

    def test_obtain_best_parameter_set(self):
        self.parameters.params["optimization"] = [
            {"param_name": ["expansion_coefficient"], "initial_point": 0, "final_point": 1, "step_size": 0.1},
            {"param_name": ["word2vec", "threshold"], "initial_point": 0, "final_point": 1, "step_size": 0.1}
        ]
        self.parameters.params["optimized_parameters_file_name"] = "test_files/optimized_parameters.json"

        self.parameters.params["word2vec"] = {
            "threshold": 0.6
        }
        self.parameters.params["expansion_coefficient"] = 0.05
        query_weights_optimizer = QueryWeightsOptimizer(self.parameters)

        query_weights_optimizer.gen_queries = MagicMock(return_value=None)
        query_weights_optimizer.evaluate_queries = MagicMock(return_value=0.25)
        eval_res_dict = query_weights_optimizer.obtain_best_parameter_set()

        self.assertEqual(eval_res_dict, 0.25)
