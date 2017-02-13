from __future__ import print_function

import copy
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath('..'))
try:
    from queries.queriesEvaluator import QueriesEvaluator
    from queries.queryLanguageModifier import QueryLanguageModifier
    from parameters.parameters import Parameters
except:
    raise

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 11 / 22 / 16


class QueryWeightsOptimizer(object):
    def __init__(self, parameters):
        self.parameters = parameters
        self.query_language_modifier = QueryLanguageModifier(self.parameters)
        self.queries_evaluator = QueriesEvaluator(self.parameters)

    def update_nested_dict(self, d, u, *keys):
        d = copy.deepcopy(d)
        keys = keys[0]
        if len(keys) > 1:
            d[keys[0]] = self.update_nested_dict(d[keys[0]], u, keys[1:])
        else:
            d[keys[0]] = u
        return d

    def gen_queries(self):
        self.query_language_modifier.run()

    def evaluate_queries(self):
        return self.queries_evaluator.run()

    @staticmethod
    def gen_test_values_offline_list(param_item):
        initial_point = param_item["initial_point"]
        final_point = param_item["final_point"]
        step_size = param_item["step_size"]
        return np.arange(initial_point, final_point + step_size, step_size)

    def obtain_best_parameter_set(self):
        self.gen_queries()
        best_eval_res = self.evaluate_queries()
        print("best_eval_res:", best_eval_res)
        for param_item in self.parameters.params["optimization"]:
            param_name = param_item["param_name"]
            for test_value in self.gen_test_values_offline_list(param_item):
                params_tmp = copy.deepcopy(self.parameters.params)
                self.parameters.params = self.update_nested_dict(self.parameters.params, test_value, param_name)
                self.gen_queries()
                eval_res = self.evaluate_queries()
                print("param_name, weights, eval_res, best_eval_res:", param_name, test_value, eval_res, best_eval_res)
                if best_eval_res >= eval_res:
                    self.parameters.params = copy.deepcopy(params_tmp)
                else:
                    best_eval_res = eval_res
                    self.parameters.write_to_parameters_file(self.parameters.params["optimized_parameters_file_name"])
        return best_eval_res

    def run(self):
        best_eval_res = self.obtain_best_parameter_set()
        return best_eval_res


if __name__ == "__main__":
    parameters_ = Parameters()
    parameters_.read_from_params_file()
    best_eval_res_ = QueryWeightsOptimizer(parameters_).run()
    print(best_eval_res_)
