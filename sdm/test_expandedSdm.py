from unittest import TestCase

from mock import MagicMock

from parameters.parameters import Parameters
from sdm.expanded_sdm import ExpandedSdm


def mock_compute_weight_sdm_unigrams(similar_unigram, unigram):
    weight_gram = {
        ('hello', 'hello'): 0.9,
        ('world', 'hello'): 0.1,
        ('how', 'how'): 0.6,
        ('are', 'how'): 0.3,
        ('you', 'how'): 0.1,
    }
    return weight_gram[(similar_unigram, unigram[0][0])]


def mock_compute_weight_sdm_unigrams_1(similar_unigram, unigram):
    weight_gram = {
        ('hello', 'hello'): 0.9,
        ('world', 'hello'): 0.1,
        ('how', 'how'): 0.6,
        ('are', 'how'): 0.0,
        ('you', 'how'): 0.1,
    }
    return weight_gram[(similar_unigram, unigram[0][0])]


def mock_compute_weight_sdm_bigrams(term, unigram_1, unigram_2, operator):
    del operator
    [similar_unigram_1, similar_unigram_2] = term.split(' ')
    weight_gram = {
        ('hello', 'hello', 'how', 'how'): 0.6,
        ('world', 'hello', 'how', 'how'): 0.2,
        ('hello', 'hello', 'are', 'how'): 0.3,
        ('world', 'hello', 'are', 'how'): 0.4,
        ('hello', 'hello', 'you', 'how'): 0.1,
        ('world', 'hello', 'you', 'how'): 0.5,
    }
    return weight_gram[(similar_unigram_1, unigram_1[0][0], similar_unigram_2, unigram_2[0][0])]


class TestExpandedSdm(TestCase):
    def setUp(self):
        self.parameters = Parameters()
        self.parameters.params["repo_dir"] = '../index/test_files/index'
        self.parameters.params['feature_parameters'] = {}
        self.parameters.params['feature_parameters']['UnorderedBigramWeights'] = {
            "uw_expression_norm_count": {
                "window_size": 17
            },
            "uw_expression_norm_document_count": {
                "window_size": 17
            }
        }
        self.parameters.params['feature_parameters']['OrderedBigramWeights'] = {
            "od_expression_norm_count": {
                "window_size": 17
            },
            "od_expression_norm_document_count": {
                "window_size": 17
            }
        }
        self.parameters.params['features_weights'] = {}
        self.parameters.params['features_weights']['UnorderedBigramWeights'] = {
            "uw_expression_norm_count": 0.33,
            "uw_expression_norm_document_count": 0.33,
            "bigrams_cosine_similarity_with_orig": 0.33
        }
        self.parameters.params['features_weights']['OrderedBigramWeights'] = {
            "od_expression_norm_count": 0.33,
            "od_expression_norm_document_count": 0.33,
            "bigrams_cosine_similarity_with_orig": 0.33
        }
        self.parameters.params['feature_parameters']['UnigramWeights'] = {
            "norm_term_count": {
            },
            "norm_document_count": {
            },
            "unigrams_cosine_similarity_with_orig": {
            }
        }
        self.parameters.params['features_weights']['UnigramWeights'] = {
            "norm_term_count": 0.33,
            "norm_document_count": 0.33,
            "unigrams_cosine_similarity_with_orig": 0.33
        }
        self.parameters.params["window_size"] = {}
        self.parameters.params["window_size"]["#od"] = 4
        self.parameters.params["window_size"]["#uw"] = 17
        self.parameters.params["expansion_coefficient"] = 0.2

    def test_gen_sdm_bigrams_field_1_text(self):
        unigrams_in_embedding_space = [[('hello', 1), ('world', 0.65)],
                                       [('how', 1), ('are', 0.8), ('you', 0.74)]]
        expanded_sdm = ExpandedSdm(self.parameters)
        expanded_sdm.compute_weight_sdm_bigrams = MagicMock(
            side_effect=mock_compute_weight_sdm_bigrams)
        res = expanded_sdm.gen_sdm_bigrams_field_1_text(unigrams_in_embedding_space,
                                                        "#uw")
        expected_res = """#weight(
0.6#uw17(hello how)
0.3#uw17(hello are)
0.1#uw17(hello you)
0.2#uw17(world how)
0.4#uw17(world are)
0.5#uw17(world you)
)
"""
        self.assertEqual(res, expected_res)

    def test_gen_sdm_unigrams_field_1_text(self):
        unigrams_in_embedding_space = [[('hello', 1), ('world', 0.65)],
                                       [('how', 1), ('are', 0.8), ('you', 0.74)]]
        expanded_sdm = ExpandedSdm(self.parameters)
        expanded_sdm.compute_weight_sdm_unigrams = MagicMock(
            side_effect=mock_compute_weight_sdm_unigrams)
        res = expanded_sdm.gen_sdm_unigrams_field_1_text(unigrams_in_embedding_space)
        expected_res = """#weight(
0.9#combine(hello)
0.1#combine(world)
0.6#combine(how)
0.3#combine(are)
0.1#combine(you)
)
"""
        self.assertEqual(res, expected_res)

        expanded_sdm.compute_weight_sdm_unigrams = MagicMock(
            side_effect=mock_compute_weight_sdm_unigrams_1)
        res = expanded_sdm.gen_sdm_unigrams_field_1_text(unigrams_in_embedding_space)
        expected_res = """#weight(
0.9#combine(hello)
0.1#combine(world)
0.6#combine(how)
0.1#combine(you)
)
"""
        self.assertEqual(res, expected_res)

    def test_gen_sdm_field_1_text(self):
        unigrams_in_embedding_space = [[('hello', 1), ('world', 0.65)],
                                       [('how', 1), ('are', 0.8), ('you', 0.74)]]
        expanded_sdm = ExpandedSdm(self.parameters)
        expanded_sdm.compute_weight_sdm_bigrams = MagicMock(
            side_effect=mock_compute_weight_sdm_bigrams)
        res = expanded_sdm.gen_sdm_field_1_text(unigrams_in_embedding_space,
                                                "#uw")
        expected_res = """#weight(
0.6#uw17(hello how)
0.3#uw17(hello are)
0.1#uw17(hello you)
0.2#uw17(world how)
0.4#uw17(world are)
0.5#uw17(world you)
)
"""
        self.assertEqual(res, expected_res)

    def test_compute_weight_sdm_unigrams(self):
        unigram_nearest_neighbor = [('hello', 1), ('world', 0.65)]

        expanded_sdm = ExpandedSdm(self.parameters)
        res = expanded_sdm.compute_weight_sdm_unigrams("world", unigram_nearest_neighbor)
        expected_res = -0.46492992497354174
        self.assertLess(abs(res - expected_res), 0.001)

    def test_compute_weight_sdm_bigrams(self):
        unigram_nearest_neighbor_1 = [('hello', 1), ('world', 0.65)]
        unigram_nearest_neighbor_2 = [('how', 1), ('are', 0.8), ('you', 0.74)]

        expanded_sdm = ExpandedSdm(self.parameters)

        res = expanded_sdm.compute_weight_sdm_bigrams("world are", unigram_nearest_neighbor_1,
                                                      unigram_nearest_neighbor_2, "#uw")
        expected_res = -0.45997992497354184
        self.assertLess(abs(res - expected_res), 0.001)
