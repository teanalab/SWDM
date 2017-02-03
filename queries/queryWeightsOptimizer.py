from __future__ import print_function

import operator
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.abspath('..'))
try:
    from queries.queriesEvaluator import QueriesEvaluator
    from queries.queryLanguageModifier import QueryLanguageModifier
except:
    raise

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 11 / 22 / 16


class QueryWeightsOptimizer(object):
    def __init__(self, step_size):
        self.step_size = step_size

    @staticmethod
    def evaluate_queries(params):
        return QueriesEvaluator().run(**params)

    @staticmethod
    def gen_queries(params_gen, field_weights):
        params_gen['field_weights'] = field_weights
        QueryLanguageModifier().run(**params_gen)

    def gen_weights_offline_list(self):
        weights_offline_list = []
        w2 = np.arange(0, 1, self.step_size)
        for w in w2:
            field_weights = {
                "SHORTABSTRACT".lower(): 1 - w,
                "IM2TXT".lower(): w
            }
            weights_offline_list += [field_weights]
        return weights_offline_list

    def obtain_evals_for_weights_offline_list(self, params_gen, params_eval, weights_offline_list):
        eval_res_dict = defaultdict()
        for weights in weights_offline_list:
            self.gen_queries(params_gen, field_weights=weights)
            eval_res = self.evaluate_queries(params_eval)
            print("weights, eval_res:", weights, eval_res)
            eval_res_dict[frozenset(weights.items())] = eval_res
        return eval_res_dict

    def maximize_evals_for_weights_offline_list(self, params_gen, params_eval, weights_offline_list):
        eval_res_dict = self.obtain_evals_for_weights_offline_list(params_gen, params_eval, weights_offline_list)
        best_weights = max(eval_res_dict.iteritems(), key=operator.itemgetter(1))[0]
        return eval_res_dict[best_weights], best_weights

    def run(self, params_gen, params_eval):
        weights_offline_list = self.gen_weights_offline_list()
        max_eval, best_weights = self.maximize_evals_for_weights_offline_list(params_gen, params_eval,
                                                                              weights_offline_list)
        return max_eval, best_weights


if __name__ == "__main__":
    params_gen_ = {"index_dir": "/scratch/index/indri_5_7/robust/",
                   "old_indri_query_file": "../configs/queries/robust04.cfg",
                   "new_indri_query_file":
                       "configs/queries/robust04_expanded.cfg",
                   "previous_runs_file": None}  # "../configs/runs/LOD_short_abstracts_for_image_extraction.dsv"}
    params_eval_ = {'indrirunquery_bin': '/home/fj9124/thirdPartyProgs/indri-5.11/build/bin/IndriRunQuery',
                    'trec_eval_bin': '/home/fj9124/projects/ir/evaluators/trec_eval/trec_eval.9.0/trec_eval',
                    'query_cfg_file': params_gen_.get("new_indri_query_file"),
                    'measure': 'map',
                    'query_set_opt': 'SemSearch_ES'}
    step_size_ = 0.01
    eval_res_, weights_ = QueryWeightsOptimizer(step_size_).run(params_gen_, params_eval_)
    print(eval_res_)
