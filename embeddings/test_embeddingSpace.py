from unittest import TestCase

import sys

from embeddings.embedding_space import EmbeddingSpace
from parameters.parameters import Parameters


class TestEmbeddingSpace(TestCase):
    def setUp(self):
        self.parameters = Parameters()
        self.parameters.params["word2vec"] = {"threshold": 0.6, "n_max": 5}
        self.parameters.params["repo_dir"] = '../index/test_files/index'

    def test_find_unigrams_in_embedding_space(self):
        embedding_space = EmbeddingSpace(self.parameters)
        embedding_space.initialize()
        unigrams = embedding_space.find_unigrams_in_embedding_space("hello world how are you")
        unigrams_expected = [[('hello', 1), ('hi', 0.6548984050750732), ('goodbye', 0.639905571937561),
                              ('howdy', 0.6310956478118896)],
                             [('world', 1), ('globe', 0.6945997476577759), ('theworld', 0.6902236938476562)],
                             [('how', 1), ('what', 0.6820360422134399)], [('are', 1), ('were', 0.7415369153022766)],
                             [('you', 1), ('your', 0.7808908224105835), ('yourself', 0.7698668241500854),
                              ('I', 0.6739810109138489), ('we', 0.6565826535224915), ('somebody', 0.6341674327850342)]]
        self.assertEqual(unigrams, unigrams_expected)

    def test_check_if_already_stem_exists(self):
        embedding_space = EmbeddingSpace(self.parameters)

        unigrams_in_embedding_space_pruned = [('you', 1), ('your', 0.7808908224105835),
                                              ('yourself', 0.7698668241500854), ('I', 0.6739810109138489),
                                              ('we', 0.6565826535224915), ('somebody', 0.6341674327850342)]
        unigram = "Where"
        orig_unigram = "You"
        res = embedding_space.check_if_already_stem_exists(unigrams_in_embedding_space_pruned, unigram, orig_unigram)
        self.assertEqual(res, False)

        unigram = "YOU"
        orig_unigram = "You"
        res = embedding_space.check_if_already_stem_exists(unigrams_in_embedding_space_pruned, unigram, orig_unigram)
        self.assertEqual(res, True)
