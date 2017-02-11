from unittest import TestCase

import sdm.weights.unigram_weights


class TestUnigramWeights(TestCase):
    def test_compute_weight(self):
        unigram_nearest_neighbor = [('hello', 1), ('world', 0.65)]

        unigram_weights = sdm.weights.unigram_weights.UnigramWeights('../../index/test_files/index')

        term_dependent_feature_parameters = {
            "unigram_nearest_neighbor": unigram_nearest_neighbor,
        }
        res = unigram_weights.compute_weight("world", term_dependent_feature_parameters)
        expected_res = 0.21450000000000002
        self.assertEqual(res, expected_res)