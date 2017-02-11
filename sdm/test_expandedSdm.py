from unittest import TestCase

from mock import MagicMock

import sdm.weights.unigram_weights
from sdm.expanded_sdm import ExpandedSdm


def mock_compute_weight_sdm_unigrams(similar_unigram, unigram):
    weight_gram = {
        ('hello', 'hello'): 0.9,
        ('world', 'hello'): 0.1,
        ('how', 'how'): 0.6,
        ('are', 'how'): 0.3,
        ('you', 'how'): 0.1,
    }
    return weight_gram[(similar_unigram[0], unigram[0][0])]


def mock_compute_weight_sdm_bigrams(similar_unigram_1, unigram_1, similar_unigram_2, unigram_2, operator):
    del operator
    weight_gram = {
        ('hello', 'hello', 'how', 'how'): 0.6,
        ('world', 'hello', 'how', 'how'): 0.2,
        ('hello', 'hello', 'are', 'how'): 0.3,
        ('world', 'hello', 'are', 'how'): 0.4,
        ('hello', 'hello', 'you', 'how'): 0.1,
        ('world', 'hello', 'you', 'how'): 0.5,
    }
    return weight_gram[(similar_unigram_1[0], unigram_1[0][0], similar_unigram_2[0], unigram_2[0][0])]


class TestExpandedSdm(TestCase):
    def test_gen_sdm_bigrams_field_1_text(self):
        unigrams_in_embedding_space = [[('hello', 1), ('world', 0.65)],
                                       [('how', 1), ('are', 0.8), ('you', 0.74)]]
        expanded_sdm = ExpandedSdm('../../index/test_files/index')
        expanded_sdm.compute_weight_sdm_bigrams = MagicMock(
            side_effect=mock_compute_weight_sdm_bigrams)
        res = expanded_sdm.gen_sdm_bigrams_field_1_text(unigrams_in_embedding_space,
                                                        "#uw")
        expected_res = """#weight(
0.6#uw(hello how)
0.3#uw(hello are)
0.1#uw(hello you)
0.2#uw(world how)
0.4#uw(world are)
0.5#uw(world you)
)
"""
        self.assertEqual(res, expected_res)

    def test_gen_sdm_unigrams_field_1_text(self):
        unigrams_in_embedding_space = [[('hello', 1), ('world', 0.65)],
                                       [('how', 1), ('are', 0.8), ('you', 0.74)]]
        expanded_sdm = ExpandedSdm('../../index/test_files/index')
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

    def test_gen_sdm_field_1_text(self):
        unigrams_in_embedding_space = [[('hello', 1), ('world', 0.65)],
                                       [('how', 1), ('are', 0.8), ('you', 0.74)]]
        expanded_sdm = ExpandedSdm('../../index/test_files/index')
        expanded_sdm.compute_weight_sdm_bigrams = MagicMock(
            side_effect=mock_compute_weight_sdm_bigrams)
        res = expanded_sdm.gen_sdm_field_1_text(unigrams_in_embedding_space,
                                                "#uw")
        expected_res = """#weight(
0.6#uw(hello how)
0.3#uw(hello are)
0.1#uw(hello you)
0.2#uw(world how)
0.4#uw(world are)
0.5#uw(world you)
)
"""
        self.assertEqual(res, expected_res)

    def test_compute_weight_sdm_unigrams(self):
        unigram_nearest_neighbor = [('hello', 1), ('world', 0.65)]

        expanded_sdm = ExpandedSdm('../index/test_files/index')
        res = expanded_sdm.compute_weight_sdm_unigrams("world", unigram_nearest_neighbor)
        expected_res = 0.21450000000000002
        self.assertEqual(res, expected_res)

    def test_compute_weight_sdm_bigrams(self):
        unigram_nearest_neighbor_1 = [('hello', 1), ('world', 0.65)]
        unigram_nearest_neighbor_2 = [('how', 1), ('are', 0.8), ('you', 0.74)]

        expanded_sdm = ExpandedSdm('../index/test_files/index')

        res = expanded_sdm.compute_weight_sdm_bigrams("world", unigram_nearest_neighbor_1, "are",
                                                      unigram_nearest_neighbor_2, "#uw")
        expected_res = 0.23925000000000005
        self.assertEqual(res, expected_res)
