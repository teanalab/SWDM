from __future__ import print_function

import os
from unittest import TestCase

import sys

from embeddings.word2vec import Word2vec


class TestWord2vec(TestCase):
    def setUp(self):
        self.word2vec = Word2vec()

    def test_gen_similar_words(self):
        self.word2vec.pre_trained_google_news_300_model()

        res = self.word2vec.gen_similar_words('human', 20)

        expected_res = [(u'human_beings', 0.613968014717102), (u'humans', 0.5917960405349731),
                        (u'impertinent_flamboyant_endearingly', 0.5868302583694458),
                        (u'employee_Laura_Althouse', 0.5639358758926392),
                        (u'humankind', 0.5636305809020996), (u'Human', 0.5524993538856506),
                        (u'mankind', 0.5346406698226929), (u'Christine_Gaugler_head', 0.5272535681724548),
                        (u'humanity', 0.5262271165847778), (u'sentient_intelligent', 0.5201493501663208),
                        (u'nonhuman', 0.5158316493034363), (u'nonhuman_animals', 0.5148903131484985),
                        (u'growth_hormone_misbranding', 0.5118885040283203), (u'nonhuman_species', 0.5054227113723755),
                        (u'tiniest_fragments', 0.5019494891166687), (u'animal', 0.4987776577472687),
                        (u'Treating_Gina', 0.4941716492176056), (u'Broadcastr_seeks', 0.4918348491191864),
                        (u'Stephen_Mimnaugh', 0.4915102422237396), (u'sentient_beings', 0.4901559352874756)]

        self.assertEqual(res, expected_res)

        res = self.word2vec.gen_similar_words('and', 20)
        self.assertEqual(res, [])
