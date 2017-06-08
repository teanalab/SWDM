import sys
import time
import unittest

from parameters.parameters import Parameters
from unigrams.embedding_space import EmbeddingSpace

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 11 / 21 / 16


class TestEmbeddingSpace(unittest.TestCase):
    def setUp(self):
        self.parameters = Parameters()
        self.parameters.params["word2vec"] = {"upper_threshold": 0.8, "lower_threshold": 0.6, "n_max": 5}
        self.parameters.params["repo_dir"] = '../index/test_files/index'

    def test_find_unigrams_in_embedding_space(self):
        self.parameters.params["word2vec"] = {"upper_threshold": 1, "lower_threshold": 0, "n_max": 5}
        embedding_space = EmbeddingSpace(self.parameters)
        embedding_space.initialize()
        unigrams = list(embedding_space.find_unigrams("hello world how are you".split(' ')))
        print(unigrams, file=sys.stderr)
        unigrams_expected = [[], [], [('much', 0.438563734292984), ('pretty', 0.4094923734664917)],
                             [('ones', 0.4666799008846283), ('must', 0.3789885640144348)],
                             [('youre', 0.6245741844177246)]]
        self.assertEqual(unigrams, unigrams_expected)

    def test_find_unigrams_in_embedding_space_1(self):
        self.parameters.params["word2vec"] = {"upper_threshold": 0.5, "lower_threshold": 0.4, "n_max": 5}
        embedding_space = EmbeddingSpace(self.parameters)
        embedding_space.initialize()
        unigrams = list(embedding_space.find_unigrams("hello world how are you".split(' ')))
        print(unigrams, file=sys.stderr)
        unigrams_expected = [[], [], [('much', 0.438563734292984), ('pretty', 0.4094923734664917)],
                             [('ones', 0.4666799008846283)], []]
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

    def test_gen_similar_words(self):
        embedding_space = EmbeddingSpace(self.parameters)
        embedding_space.initialize()

        t = time.process_time()
        res = embedding_space._gen_similar_words('hello', embedding_space.word2vec)
        print(res, file=sys.stderr)
        elapsed_time_1 = time.process_time() - t

        expected_res_trunc = [('hi', 0.6548983454704285), ('goodbye', 0.639905571937561), ('howdy', 0.6310957670211792),
                              ('goodnight', 0.5920578241348267), ('greeting', 0.5855878591537476),
                              ('Hello', 0.5842196345329285), ("g'day", 0.5754077434539795),
                              ('See_ya', 0.5688871145248413), ('ya_doin', 0.5643119812011719),
                              ('greet', 0.5636603832244873), ('hullo', 0.5621640682220459),
                              ('hellos', 0.5596432685852051), ('Hey', 0.5594544410705566),
                              ('bye_bye', 0.5593388676643372), ('bonjour', 0.5587834715843201),
                              ('adios', 0.5560759902000427), ('ciao', 0.5548770427703857), ('hug', 0.5544619560241699),
                              ('buh_bye', 0.5511860847473145), ("G'day", 0.5494421124458313)]

        self.assertEqual(res[:20], expected_res_trunc)

        t = time.process_time()
        res = embedding_space._gen_similar_words('hello', embedding_space.word2vec)
        print(res, file=sys.stderr)
        elapsed_time_2 = time.process_time() - t

        self.assertEqual(res[:20], expected_res_trunc)

        self.assertLess(elapsed_time_2, 1)
        self.assertLess(elapsed_time_2, elapsed_time_1)

    def test_check_if_unigram_should_be_added(self):
        self.parameters.params["word2vec"] = {"upper_threshold": 0.8, "lower_threshold": 0.6, "n_max": 5}
        embedding_space = EmbeddingSpace(self.parameters)
        res = embedding_space.check_if_unigram_should_be_added("planemaker", 0.75, [], "Airbus")
        self.assertFalse(res)

        self.parameters.params["word2vec"] = {"upper_threshold": 0.8, "lower_threshold": 0.6, "n_max": 5}
        embedding_space = EmbeddingSpace(self.parameters)
        res = embedding_space.check_if_unigram_should_be_added("civil", 0.6548983454704285, [], "yours")
        self.assertTrue(res)

        self.parameters.params["word2vec"] = {"upper_threshold": 0.8, "lower_threshold": 0.7, "n_max": 5}
        embedding_space = EmbeddingSpace(self.parameters)
        res = embedding_space.check_if_unigram_should_be_added("civil", 0.6548983454704285, [], "yours")
        self.assertFalse(res)

        self.parameters.params["word2vec"] = {"upper_threshold": 0.6, "lower_threshold": 0.5, "n_max": 5}
        embedding_space = EmbeddingSpace(self.parameters)
        res = embedding_space.check_if_unigram_should_be_added("civil", 0.6548983454704285, [], "yours")
        self.assertFalse(res)


if __name__ == '__main__':
    unittest.main()
