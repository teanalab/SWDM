from unittest import TestCase

import sdm.weights.ordered_bigram_weights


class TestOrderedBigramWeights(TestCase):
    def test_compute_weight(self):
        unigram_nearest_neighbor_1 = [('hello', 1), ('world', 0.65)]
        unigram_nearest_neighbor_2 = [('how', 1), ('are', 0.8), ('you', 0.74)]

        ordered_bigram_weights = sdm.weights.ordered_bigram_weights.OrderedBigramWeights(
            '../../index/test_files/index')

        term_dependent_feature_parameters = {
            "unigram_nearest_neighbor_1": unigram_nearest_neighbor_1,
            "unigram_nearest_neighbor_2": unigram_nearest_neighbor_2
        }
        res = ordered_bigram_weights.compute_weight("world are", term_dependent_feature_parameters)
        expected_res = 0.23925000000000005
        self.assertEqual(res, expected_res)
