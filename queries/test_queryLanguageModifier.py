from unittest import TestCase

import sys

from queries.queryLanguageModifier import QueryLanguageModifier

from mock import MagicMock


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


class TestQueryLanguageModifier(TestCase):
    def setUp(self):
        pass

    def test_gen_combine_fields_text(self):
        field_weights = {
            "u": 0.8,
            "o": 0.1,
            "w": 0.1
        }
        field_texts = {'u': '#weight(\n0.1 #combine(hello)\n0.2 #combine(world)\n0.3 #combine(how)\n'
                            '0.4 #combine(are)\n0.5 #combine(you)\n)\n',
                       'w': '#weight(\n0.1 #uw17(hello world)\n0.2 #uw17(world how)\n0.3 #uw17(how are)\n'
                            '0.4 #uw17(are you)\n)\n',
                       'o': '#weight(\n0.1 #od4(hello world)\n0.2 #od4(world how)\n0.3 #od4(how are)\n'
                            '0.4 #od4(are you)\n)\n'}

        query_language_modifier = QueryLanguageModifier()
        res = query_language_modifier.gen_combine_fields_text(field_weights, field_texts)
        print(res, file=sys.stderr)
        expected_res = "#weight(\n" \
                       "0.8 #weight(\n" \
                       "0.1 #combine(hello)\n" \
                       "0.2 #combine(world)\n" \
                       "0.3 #combine(how)\n" \
                       "0.4 #combine(are)\n" \
                       "0.5 #combine(you)\n" \
                       ")\n" \
                       "0.1 #weight(\n" \
                       "0.1 #od4(hello world)\n" \
                       "0.2 #od4(world how)\n" \
                       "0.3 #od4(how are)\n" \
                       "0.4 #od4(are you)\n" \
                       ")\n" \
                       "0.1 #weight(\n" \
                       "0.1 #uw17(hello world)\n" \
                       "0.2 #uw17(world how)\n" \
                       "0.3 #uw17(how are)\n" \
                       "0.4 #uw17(are you)\n" \
                       ")\n" \
                       ")\n"
        self.assertEqual(res, expected_res)

    def test_gen_sdm_field_1_text(self):
        grams = {
            'hello world',
            'world how',
            'how are',
            'are you'
        }
        query_language_modifier = QueryLanguageModifier()
        query_language_modifier.compute_weight_sdm_unigrams = MagicMock(side_effect=mock_compute_weight_sdm_unigrams)
        res = query_language_modifier.gen_sdm_field_1_text(grams, 2, "#uw17")
        expected_res = "#weight(\n" \
                       "0.1 #uw17(hello world)\n" \
                       "0.3 #uw17(how are)\n" \
                       "0.4 #uw17(are you)\n" \
                       "0.2 #uw17(world how)\n" \
                       ")\n"
        self.assertEqual(res, expected_res)

    def test_gen_sdm_fields_texts(self):
        query_language_modifier = QueryLanguageModifier()
        query_language_modifier.compute_weight_sdm_unigrams = MagicMock(side_effect=mock_compute_weight_sdm_unigrams)
        query_language_modifier.compute_weight_sdm_bigrams = MagicMock(side_effect=mock_compute_weight_sdm_bigrams)
        query_language_modifier.find_unigrams_in_embedding_space = MagicMock(return_value=
                                                                             [[('hello', 1), ('world', 0.65)],
                                                                              [('how', 1), ('are', 0.8),
                                                                               ('you', 0.74)]])
        res = query_language_modifier.gen_sdm_fields_texts("hello world how are you")
        print(res, file=sys.stderr)
        expected_res = {'u': '#weight(\n0.9 #combine(hello)\n0.1 #combine(world)\n0.6 #combine(how)\n'
                             '0.3 #combine(are)\n0.1 #combine(you)\n)\n',
                        'w': '#weight(\n0.6 #uw17(hello how)\n0.3 #uw17(hello are)\n0.1 #uw17(hello you)\n'
                             '0.2 #uw17(world how)\n0.4 #uw17(world are)\n0.5 #uw17(world you)\n)\n',
                        'o': '#weight(\n0.6 #od4(hello how)\n0.3 #od4(hello are)\n0.1 #od4(hello you)\n'
                             '0.2 #od4(world how)\n0.4 #od4(world are)\n0.5 #od4(world you)\n)\n'}

        self.assertEqual(res, expected_res)

    def test_find_unigrams_in_embedding_space(self):
        query_language_modifier = QueryLanguageModifier()
        unigrams = query_language_modifier.find_unigrams_in_embedding_space("hello world how are you")
        unigrams_expected = [[('hello', 1), ('hi', 0.6548984050750732), ('goodbye', 0.639905571937561),
                              ('howdy', 0.6310956478118896)],
                             [('world', 1), ('globe', 0.6945997476577759), ('theworld', 0.6902236938476562)],
                             [('how', 1), ('what', 0.6820360422134399), ('How', 0.6297600865364075)],
                             [('are', 1), ('were', 0.7415369153022766), ('Are', 0.6810149550437927),
                              ("'re", 0.6806748509407043), ('aren_t', 0.6020350456237793)],
                             [('you', 1), ('You', 0.8077561259269714), ('your', 0.7808908224105835),
                              ('yourself', 0.7698668241500854), ('I', 0.6739810109138489),
                              ('we', 0.6565826535224915), ("Don'tI", 0.6465235352516174),
                              ('somebody', 0.6341674327850342), ('yours', 0.6339222192764282),
                              ('ifyou', 0.6254765391349792), ('youre', 0.6245741844177246),
                              ("Ican't", 0.623076319694519), ("Can'tI", 0.6217659711837769),
                              ("your're", 0.6169878244400024), ('QDo', 0.6142182946205139),
                              ("'ll", 0.6046315431594849), ("don'tI", 0.6028058528900146),
                              ('youwant', 0.6026159524917603), ('youcan', 0.6023932695388794)]]

        self.assertEqual(unigrams, unigrams_expected)

    def test_gen_sdm_unigrams_field_1_text(self):
        unigrams_in_embedding_space = [[('hello', 1), ('world', 0.65)],
                                       [('how', 1), ('are', 0.8), ('you', 0.74)]]
        query_language_modifier = QueryLanguageModifier()
        query_language_modifier.compute_weight_sdm_unigrams = MagicMock(side_effect=mock_compute_weight_sdm_unigrams)
        res = query_language_modifier.gen_sdm_unigrams_field_1_text(unigrams_in_embedding_space, "#combine")
        expected_res = "#weight(\n" \
                       "0.9 #combine(hello)\n" \
                       "0.1 #combine(world)\n" \
                       "0.6 #combine(how)\n" \
                       "0.3 #combine(are)\n" \
                       "0.1 #combine(you)\n" \
                       ")\n"
        self.assertEqual(res, expected_res)
