from unittest import TestCase

import sdm.weights.ordered_bigram_weights
from parameters.parameters import Parameters

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 11 / 21 / 16


class TestOrderedBigramWeights(TestCase):
    def setUp(self):
        self.parameters = Parameters()
        self.parameters.params["repo_dir"] = '../../index/test_files/index'
        self.parameters.params["type_weights"]["exp_embed"] = 0.05
        self.parameters.params["type_weights"]["exp_top_docs"] = 0
        self.parameters.params['feature_parameters'] = {}
        self.parameters.params['feature_parameters']['OrderedBigramWeights'] = {
            "od_expression_norm_count": {
                "window_size": 4
            },
            "od_expression_norm_document_count": {
                "window_size": 4
            },
            "td_od_expression_norm_count": {
                "window_size": 17,
                "n_top_docs": 10,
            },
            "td_od_expression_norm_document_count": {
                "window_size": 17,
                "n_top_docs": 10,
            },
        }
        self.parameters.params['features_weights'] = {}
        self.parameters.params['features_weights']['OrderedBigramWeights'] = {
            "od_expression_norm_count": 0.33,
            "od_expression_norm_document_count": 0.33,
            "bigrams_cosine_similarity_with_orig": 0.33,
            "td_od_expression_norm_count": 0.33,
            "td_od_expression_norm_document_count": 0.33,
        }

    def test_compute_weight(self):
        unigram_nearest_neighbor_1 = [('hello', 1), ('world', 0.65)]
        unigram_nearest_neighbor_2 = [('how', 1), ('are', 0.8), ('you', 0.74)]

        ordered_bigram_weights = sdm.weights.ordered_bigram_weights.OrderedBigramWeights(self.parameters)
        ordered_bigram_weights.init_top_docs_run_query("a")

        term_dependent_feature_parameters = {
            "unigram_nearest_neighbor_1": unigram_nearest_neighbor_1,
            "unigram_nearest_neighbor_2": unigram_nearest_neighbor_2
        }
        res = ordered_bigram_weights.compute_weight("world are", term_dependent_feature_parameters)
        expected_res = 0.5714860275162977
        self.assertEqual(res, expected_res)
