from __future__ import print_function

from unittest import TestCase

from queryLanguageModifier import QueryLanguageModifier


class TestQueryLanguageModifier(TestCase):
    def setUp(self):
        self.queryLanguageModifier = QueryLanguageModifier()

    def test_get_field_texts(self):
        res = self.queryLanguageModifier.get_field_texts("hello world how are you")
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
        field_texts = {'u': '#combine(hello world how are you)',
                       'w': '#combine( #uw17(hello world) #uw17(world how) #uw17(how are) #uw17(are you) )',
                       'o': '#combine( #od4(hello world) #od4(world how) #od4(how are) #od4(are you) )'}

        res = self.queryLanguageModifier.gen_combine_fields_text(field_weights, field_texts)
        expected_res = '\n   #weight(\n      0.8#combine(hello world how are you)\n      ' + \
                       '0.1#combine( #uw17(hello world) #uw17(world how) #uw17(how are) #uw17(are you) )\n      ' + \
                       '0.1#combine( #od4(hello world) #od4(world how) #od4(how are) #od4(are you) )\n   )\n'


        self.assertEqual(res, expected_res)
