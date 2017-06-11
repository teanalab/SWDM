from unittest import TestCase

import features.embeddings

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 11 / 21 / 16


class TestEmbeddings(TestCase):
    def test_cosine_similarity_with_orig(self):
        feature_parameters = {}
        unigram_pair = ('two', ('pair', 0.5850129127502441))
        res = features.embeddings.Embeddings().unigrams_cosine_similarity_with_orig(unigram_pair, feature_parameters)
        expected_res = 0.5850129127502441

        self.assertEqual(res, expected_res)

    def test_bigrams_cosine_similarity_with_orig(self):
        feature_parameters = {}
        bigram_pair = (('two', 'stand'), (('pair', 0.5850129127502441), ('sit', 0.5178298950195312)))

        res = features.embeddings.Embeddings().bigrams_cosine_similarity_with_orig(bigram_pair, feature_parameters)
        expected_res = 0.5514214038848877

        self.assertEqual(res, expected_res)
