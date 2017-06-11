from unittest import TestCase

import sdm.weights.unigram_weights
from parameters.parameters import Parameters

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 11 / 21 / 16


class TestUnigramWeights(TestCase):
    def setUp(self):
        self.parameters = Parameters()
        self.parameters.params["repo_dir"] = '../../index/test_files/index'
        self.parameters.params["type_weights"] = {"exp_embed": 0.05,
                                                  "exp_top_docs": 0}
        self.parameters.params['feature_parameters'] = {}
        self.parameters.params['feature_parameters']['UnigramWeights'] = {
            "norm_term_count": {
            },
            "norm_document_count": {
            },
            "unigrams_cosine_similarity_with_orig": {
            },
            "td_norm_unigram_count": {
                "n_top_docs": 10
            },
            "td_norm_unigram_document_count": {
                "n_top_docs": 10
            },
        }
        self.parameters.params['features_weights'] = {}
        self.parameters.params['features_weights']['UnigramWeights'] = {
            "norm_term_count": 0.33,
            "norm_document_count": 0.33,
            "unigrams_cosine_similarity_with_orig": 0.33,
            "td_norm_unigram_count": 0.33,
            "td_norm_unigram_document_count": 0.33,
        }

    def test_compute_weight(self):
        unigram_weights = sdm.weights.unigram_weights.UnigramWeights(self.parameters)
        unigram_weights.init_top_docs_run_query("a")

        unigram_pair = ('two', ('pair', 0.5850129127502441))
        res = unigram_weights.compute_weight(unigram_pair)
        expected_res = 4.75371025803143
        self.assertAlmostEquals(res, expected_res)
