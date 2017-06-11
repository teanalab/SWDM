from unittest import TestCase

import sdm.weights.unordered_bigram_weights
from parameters.parameters import Parameters

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 11 / 21 / 16


class TestUnorderedBigramWeights(TestCase):
    def setUp(self):
        self.parameters = Parameters()
        self.parameters.params["repo_dir"] = '../../index/test_files/index'
        self.parameters.params["type_weights"]["exp_embed"] = 0.05
        self.parameters.params["type_weights"]["exp_top_docs"] = 0
        self.parameters.params['feature_parameters'] = {}
        self.parameters.params['feature_parameters']['UnorderedBigramWeights'] = {
            "uw_expression_norm_count": {
                "window_size": 17,
            },
            "uw_expression_norm_document_count": {
                "window_size": 17,
            },
            "td_uw_expression_norm_count": {
                "window_size": 17,
                "n_top_docs": 10,
            },
            "td_uw_expression_norm_document_count": {
                "window_size": 17,
                "n_top_docs": 10,
            },
        }
        self.parameters.params['features_weights'] = {}
        self.parameters.params['features_weights']['UnorderedBigramWeights'] = {
            "uw_expression_norm_count": 0.33,
            "uw_expression_norm_document_count": 0.33,
            "bigrams_cosine_similarity_with_orig": 0.33,
            "td_uw_expression_norm_count": 0.33,
            "td_uw_expression_norm_document_count": 0.33,
        }

    def test_compute_weight(self):
        unordered_bigram_weights = sdm.weights.unordered_bigram_weights.UnorderedBigramWeights(self.parameters)
        unordered_bigram_weights.init_top_docs_run_query("a")

        res = unordered_bigram_weights.compute_weight("world are")
        expected_res = 0.5714860275162977
        self.assertEqual(res, expected_res)
