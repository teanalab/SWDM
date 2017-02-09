from unittest import TestCase

import features.embeddings


class TestEmbeddings(TestCase):
    def test_cosine_similarity_with_orig(self):
        feature_parameters = {
            'unigram_nearest_neighbor': [('hello', 1), ('world', 0.65)]
        }

        res = features.embeddings.Embeddings.unigrams_cosine_similarity_with_orig('world', feature_parameters)
        expected_res = 0.65

        self.assertEqual(res, expected_res)

    def test_bigrams_cosine_similarity_with_orig(self):
        feature_parameters = {
            'unigram_nearest_neighbor_1': [('hello', 1), ('world', 0.65)],
            'unigram_nearest_neighbor_2': [('how', 1), ('are', 0.8), ('you', 0.74)]
        }

        res = features.embeddings.Embeddings.bigrams_cosine_similarity_with_orig('world you', feature_parameters)
        expected_res = 0.6950000000000001

        self.assertEqual(res, expected_res)
