from unittest import TestCase

from mock import MagicMock

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


def mock_compute_weight_sdm_bigrams(similar_unigram_1, unigram_1, similar_unigram_2, unigram_2):
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
        expanded_sdm = ExpandedSdm()
        expanded_sdm.compute_weight_sdm_bigrams = MagicMock(
            side_effect=mock_compute_weight_sdm_bigrams)
        res = expanded_sdm.gen_sdm_bigrams_field_1_text(unigrams_in_embedding_space,
                                                        "#uw")
        expected_res = """#weight(
0.6 #uw(hello how)
0.3 #uw(hello are)
0.1 #uw(hello you)
0.2 #uw(world how)
0.4 #uw(world are)
0.5 #uw(world you)
)
"""
        self.assertEqual(res, expected_res)

    def test_gen_sdm_unigrams_field_1_text(self):
        unigrams_in_embedding_space = [[('hello', 1), ('world', 0.65)],
                                       [('how', 1), ('are', 0.8), ('you', 0.74)]]
        expanded_sdm = ExpandedSdm()
        expanded_sdm.compute_weight_sdm_unigrams = MagicMock(
            side_effect=mock_compute_weight_sdm_unigrams)
        res = expanded_sdm.gen_sdm_unigrams_field_1_text(unigrams_in_embedding_space,
                                                         "#combine")
        expected_res = "#weight(\n" \
                       "0.9 #combine(hello)\n" \
                       "0.1 #combine(world)\n" \
                       "0.6 #combine(how)\n" \
                       "0.3 #combine(are)\n" \
                       "0.1 #combine(you)\n" \
                       ")\n"
        self.assertEqual(res, expected_res)
