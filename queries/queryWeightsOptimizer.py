import copy
import os
import random
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

    def update_params_nested_dict(self, d, u, *keys):
        if isinstance(u, np.int64):
            u = int(u)
        elif isinstance(u, np.float64):
            u = float(u)
        d = copy.deepcopy(d)
        keys = keys[0]
        if len(keys) > 1:
            d[keys[0]] = self.update_params_nested_dict(d[keys[0]], u, keys[1:])
        else:
            d[keys[0]] = u
        return d

    def gen_queries(self, is_test):
        self.query_language_modifier.run_no_word2vec_initialization(is_test)

    def evaluate_queries(self):
        return self.queries_evaluator.run()

    @staticmethod
    def gen_test_values_offline_list(param_item):
        initial_point = param_item["initial_point"]
        final_point = param_item["final_point"]
        step_size = param_item["step_size"]
        return np.arange(initial_point, final_point, step_size)

    def reset_params_to_previous_best(self, params_previous_best_tmp):
        self.parameters.params = copy.deepcopy(params_previous_best_tmp)

    def update_shared_param_item(self, shared_param_items, test_value):
        for param_name in shared_param_items:
            self.parameters.params = self.update_params_nested_dict(self.parameters.params, test_value, param_name)

    def examine_a_test_value(self, test_value, best_eval_res, shared_param_items_first, shared_param_items):
        params_previous_best_tmp = copy.deepcopy(self.parameters.params)
        self.update_shared_param_item(shared_param_items, test_value)
        self.gen_queries(is_test=False)
        eval_res = self.evaluate_queries()
        print("param_name, weights, eval_res, best_eval_res:", shared_param_items_first,
              test_value, eval_res, best_eval_res)
        if eval_res <= best_eval_res:
            self.reset_params_to_previous_best(params_previous_best_tmp)
        else:
            best_eval_res = eval_res
            self.parameters.write_to_parameters_file(self.parameters.params["optimized_parameters_file_name"])
        return best_eval_res, eval_res

    def find_param_optimization_item(self, shared_param_names_first):
        return next((item for item in self.parameters.params["optimization"] if item["param_name"] ==
                     shared_param_names_first))

    @staticmethod
    def check_the_progress(eval_res_history):
        if len(eval_res_history) > 2:
            if eval_res_history[-1] < eval_res_history[-2] < eval_res_history[-3]:
                return False
        return True

    def obtain_best_parameter_set(self):
        self.query_language_modifier.run(is_test=False)
        best_eval_res = self.evaluate_queries()
        print("initial best_eval_res:", best_eval_res)
        self.run_cv_tests()
        shared_params_optimization = self.parameters.params["shared_params_optimization"]
        random.shuffle(shared_params_optimization)
        for shared_param_items in shared_params_optimization:
            shared_param_names_first = shared_param_items[0]
            shared_param_items_first = self.find_param_optimization_item(shared_param_names_first)
            eval_res_history = []
            for test_value in self.gen_test_values_offline_list(shared_param_items_first):
                print("shared_param_items_first, test_value:", shared_param_names_first, test_value)
                best_eval_res, eval_res = self.examine_a_test_value(test_value, best_eval_res, shared_param_names_first,
                                                                    shared_param_items)
                eval_res_history += [eval_res]
                if not self.check_the_progress(eval_res_history):
                    break

            self.run_cv_tests()
        return best_eval_res

    def run_cv_tests(self):
        self.gen_queries(is_test=True)
        test_eval_res = self.evaluate_queries()
        print("test_eval_res:", test_eval_res)

        params_previous_best_tmp = copy.deepcopy(self.parameters.params)
        self.parameters.params = self.update_params_nested_dict(self.parameters.params, 0, ["expansion_coefficient"])
        self.gen_queries(is_test=True)
        test_eval_res = self.evaluate_queries()
        self.reset_params_to_previous_best(params_previous_best_tmp)
        print("sdm_eval_res:", test_eval_res)

    def run(self):
        best_eval_res = self.obtain_best_parameter_set()
        return best_eval_res


if __name__ == "__main__":
    parameters_ = Parameters()
    parameters_.read_from_params_file()
    best_eval_res_ = QueryWeightsOptimizer(parameters_).run()
    print(best_eval_res_)
