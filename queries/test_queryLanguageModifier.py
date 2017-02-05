from __future__ import print_function

from unittest import TestCase

import sys

from queryLanguageModifier import QueryLanguageModifier

from mock import MagicMock


def mock_compute_weight(gram):
    weight_gram = {
        'hello': 0.1,
        'world': 0.2,
        'how': 0.3,
        'are': 0.4,
        'you': 0.5,
        'hello world': 0.1,
        'world how': 0.2,
        'how are': 0.3,
        'are you': 0.4
    }
    return weight_gram.get(gram)


class TestQueryLanguageModifier(TestCase):
    def setUp(self):
        self.queryLanguageModifier = QueryLanguageModifier()

    def test_get_field_texts(self):
        res = self.queryLanguageModifier.gen_sdm_fields_texts("hello world how are you")
        expected_res = {'u': '#combine(hello world how are you)',
                        'w': '#combine( #uw17(hello world) #uw17(world how) #uw17(how are) #uw17(are you) )',
                        'o': '#combine( #od4(hello world) #od4(world how) #od4(how are) #od4(are you) )'}

        self.assertEqual(res, expected_res)

    def test_get_bigrams_list(self):
        res = self.queryLanguageModifier.get_bigrams_list("hello world how are you")
        expected_res = ['hello world', 'world how', 'how are', 'are you']
        self.assertEqual(res, expected_res)

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

        res = self.queryLanguageModifier.gen_combine_fields_text(field_weights, field_texts)
        expected_res = '#weight(\n' \
                       '0.8 #weight(\n' \
                       '0.1 #combine(hello)\n' \
                       '0.2 #combine(world)\n' \
                       '0.3 #combine(how)\n' \
                       '0.4 #combine(are)\n' \
                       '0.5 #combine(you)\n' \
                       ')\n' \
                       '0.1 #weight(\n' \
                       '0.1 #uw17(hello world)\n' \
                       '0.2 #uw17(world how)\n' \
                       '0.3 #uw17(how are)\n' \
                       '0.4 #uw17(are you)\n' \
                       ')\n' \
                       '0.1 #weight(\n' \
                       '0.1 #od4(hello world)\n' \
                       '0.2 #od4(world how)\n' \
                       '0.3 #od4(how are)\n' \
                       '0.4 #od4(are you)\n' \
                       ')\n' \
                       ')\n'

        self.assertEqual(res, expected_res)

    def test_gen_sdm_field_1_text(self):
        grams = {
            'hello world',
            'world how',
            'how are',
            'are you'
        }
        query_language_modifier = QueryLanguageModifier()
        query_language_modifier.compute_weight = MagicMock(side_effect=mock_compute_weight)
        res = query_language_modifier.gen_sdm_field_1_text(grams, "#uw17")
        expected_res = "#weight(\n" \
                       "0.1 #uw17(hello world)\n" \
                       "0.3 #uw17(how are)\n" \
                       "0.4 #uw17(are you)\n" \
                       "0.2 #uw17(world how)\n" \
                       ")\n"
        self.assertEqual(res, expected_res)

    def test_gen_sdm_fields_texts(self):
        query_language_modifier = QueryLanguageModifier()
        query_language_modifier.compute_weight = MagicMock(side_effect=mock_compute_weight)
        res = query_language_modifier.gen_sdm_fields_texts("hello world how are you")
        expected_res = {'u': '#weight(\n0.1 #combine(hello)\n0.2 #combine(world)\n0.3 #combine(how)\n'
                             '0.4 #combine(are)\n0.5 #combine(you)\n)\n',
                        'w': '#weight(\n0.1 #uw17(hello world)\n0.2 #uw17(world how)\n0.3 #uw17(how are)\n'
                             '0.4 #uw17(are you)\n)\n',
                        'o': '#weight(\n0.1 #od4(hello world)\n0.2 #od4(world how)\n0.3 #od4(how are)\n'
                             '0.4 #od4(are you)\n)\n'}
        self.assertEqual(res, expected_res)
