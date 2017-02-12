from unittest import TestCase

from mock import MagicMock

from parameters.parameters import Parameters
from queries.queryLanguageModifier import QueryLanguageModifier


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


class TestQueryLanguageModifier(TestCase):
    def setUp(self):
        self.parameters = Parameters()
        self.parameters.params["repo_dir"] = '../index/test_files/index'
        self.parameters.params["window_size"] = {}
        self.parameters.params["window_size"]["#od"] = 4
        self.parameters.params["window_size"]["#uw"] = 17

    def test_gen_combine_fields_text(self):
        field_weights = {
            "u": 0.8,
            "o": 0.1,
            "w": 0.1
        }
        field_texts = {'u': '#weight(\n0.1#combine(hello)\n0.2#combine(world)\n0.3#combine(how)\n'
                            '0.4#combine(are)\n0.5#combine(you)\n)\n',
                       'w': '#weight(\n0.1#uw17(hello world)\n0.2#uw17(world how)\n0.3#uw17(how are)\n'
                            '0.4#uw17(are you)\n)\n',
                       'o': '#weight(\n0.1#od4(hello world)\n0.2#od4(world how)\n0.3#od4(how are)\n'
                            '0.4#od4(are you)\n)\n'}

        query_language_modifier = QueryLanguageModifier(self.parameters)
        res = query_language_modifier.gen_combine_fields_text(field_weights, field_texts)
        expected_res = "#weight(\n" \
                       "0.8#weight(\n" \
                       "0.1#combine(hello)\n" \
                       "0.2#combine(world)\n" \
                       "0.3#combine(how)\n" \
                       "0.4#combine(are)\n" \
                       "0.5#combine(you)\n" \
                       ")\n" \
                       "0.1#weight(\n" \
                       "0.1#od4(hello world)\n" \
                       "0.2#od4(world how)\n" \
                       "0.3#od4(how are)\n" \
                       "0.4#od4(are you)\n" \
                       ")\n" \
                       "0.1#weight(\n" \
                       "0.1#uw17(hello world)\n" \
                       "0.2#uw17(world how)\n" \
                       "0.3#uw17(how are)\n" \
                       "0.4#uw17(are you)\n" \
                       ")\n" \
                       ")\n"
        self.assertEqual(res, expected_res)

    def test_gen_sdm_fields_texts(self):
        query_language_modifier = QueryLanguageModifier(self.parameters)
        query_language_modifier.expanded_sdm.compute_weight_sdm_unigrams = MagicMock(
            side_effect=mock_compute_weight_sdm_unigrams)
        query_language_modifier.expanded_sdm.compute_weight_sdm_bigrams = MagicMock(
            side_effect=mock_compute_weight_sdm_bigrams)
        query_language_modifier.embedding_space.find_unigrams_in_embedding_space = \
            MagicMock(return_value=[[('hello', 1),
                                     ('world', 0.65)],
                                    [('how', 1), ('are', 0.8),
                                     ('you', 0.74)]])
        res = query_language_modifier.gen_sdm_fields_texts("hello world how are you")
        expected_res = {'u': '#weight(\n0.9#combine(hello)\n0.1#combine(world)\n0.6#combine(how)\n0.3#combine(are)\n'
                             '0.1#combine(you)\n)\n',
                        'o': '#weight(\n0.6#od4(hello how)\n0.3#od4(hello are)\n0.1#od4(hello you)\n'
                             '0.2#od4(world how)\n0.4#od4(world are)\n0.5#od4(world you)\n)\n',
                        'w': '#weight(\n0.6#uw17(hello how)\n0.3#uw17(hello are)\n0.1#uw17(hello you)\n'
                             '0.2#uw17(world how)\n0.4#uw17(world are)\n0.5#uw17(world you)\n)\n'}
        self.assertEqual(res, expected_res)
