from unittest import TestCase

import sdm.weights.unigram_weights
from parameters.parameters import Parameters


class TestUnigramWeights(TestCase):
    def setUp(self):
        self.parameters = Parameters()
        self.parameters.params["repo_dir"] = '../../index/test_files/index'
        self.parameters.params['expansion_coefficient'] = 0.1
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
        unigram_nearest_neighbor = [('hello', 1), ('world', 0.65)]

        unigram_weights = sdm.weights.unigram_weights.UnigramWeights(self.parameters)
        unigram_weights.init_top_docs_run_query("a")

        term_dependent_feature_parameters = {
            "unigram_nearest_neighbor": unigram_nearest_neighbor,
        }
        res = unigram_weights.compute_weight("world", term_dependent_feature_parameters)
        expected_res = 0.5690110275162977
        self.assertEqual(res, expected_res)
