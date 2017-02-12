from unittest import TestCase

import sys

from embeddings.embedding_space import EmbeddingSpace


class TestEmbeddingSpace(TestCase):
    def test_find_unigrams_in_embedding_space(self):
        embedding_space = EmbeddingSpace()
        embedding_space.initialize()
        word2vec_threshold = 0.6
        unigrams = embedding_space.find_unigrams_in_embedding_space("hello world how are you", word2vec_threshold)
        print(unigrams, file=sys.stderr)
        unigrams_expected = [[('hello', 1), ('hi', 0.6548984050750732), ('goodbye', 0.639905571937561),
                              ('howdy', 0.6310956478118896)],
                             [('world', 1), ('globe', 0.6945997476577759), ('theworld', 0.6902236938476562)],
                             [('how', 1), ('what', 0.6820360422134399), ('How', 0.6297600865364075)],
                             [('are', 1), ('were', 0.7415369153022766), ('Are', 0.6810149550437927)],
                             [('you', 1), ('You', 0.8077561259269714), ('your', 0.7808908224105835),
                              ('yourself', 0.7698668241500854), ('I', 0.6739810109138489), ('we', 0.6565826535224915),
                              ('somebody', 0.6341674327850342), ('yours', 0.6339222192764282),
                              ('ifyou', 0.6254765391349792), ('youre', 0.6245741844177246),
                              ('QDo', 0.6142182946205139), ('youwant', 0.6026159524917603),
                              ('youcan', 0.6023932695388794)]]

        self.assertEqual(unigrams, unigrams_expected)
