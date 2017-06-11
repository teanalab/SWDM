import os
import sys
import unittest

from mock import MagicMock

sys.path.insert(0, os.path.abspath('..'))
try:
    from parameters.parameters import Parameters
    from queries.queries import Queries
    from queries.queryLanguageModifier import QueryLanguageModifier
    from sdm.test_expandedSdm import mock_compute_weight_sdm_grams
except:
    raise

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 11 / 21 / 16


class TestQueryLanguageModifier(unittest.TestCase):
    def setUp(self):
        self.parameters = Parameters()
        self.parameters.params["repo_dir"] = '../index/test_files/index'
        self.parameters.params["word2vec"] = {}
        self.parameters.params["word2vec"]["threshold"] = 0.6
        self.parameters.params["window_size"] = {}
        self.parameters.params["window_size"]["#od"] = 4
        self.parameters.params["window_size"]["#uw"] = 17
        self.parameters.params['feature_parameters'] = {}
        self.parameters.params['feature_parameters']['UnigramWeights'] = {
            "norm_term_count": {
            },
            "norm_document_count": {
            },
            "unigrams_cosine_similarity_with_orig": {
            }
        }
        self.parameters.params['feature_parameters']['UnorderedBigramWeights'] = {
            "uw_expression_norm_count": {
                "window_size": 17
            },
            "uw_expression_norm_document_count": {
                "window_size": 17
            }
        }
        self.parameters.params['feature_parameters']['OrderedBigramWeights'] = {
            "od_expression_norm_count": {
                "window_size": 17
            },
            "od_expression_norm_document_count": {
                "window_size": 17
            }
        }
        self.parameters.params['features_weights'] = {}
        self.parameters.params['features_weights']['UnigramWeights'] = {
            "norm_term_count": 0.33,
            "norm_document_count": 0.33,
            "unigrams_cosine_similarity_with_orig": 0.33
        }
        self.parameters.params['features_weights']['UnorderedBigramWeights'] = {
            "uw_expression_norm_count": 0.33,
            "uw_expression_norm_document_count": 0.33,
            "bigrams_cosine_similarity_with_orig": 0.33
        }
        self.parameters.params['features_weights']['OrderedBigramWeights'] = {
            "od_expression_norm_count": 0.33,
            "od_expression_norm_document_count": 0.33,
            "bigrams_cosine_similarity_with_orig": 0.33
        }
        self.parameters.params["word2vec"] = {"threshold": 0.6, "n_max": 5}
        self.parameters.params["type_weights"] = {
            "orig": 1,
            "exp_embed": 0.065,
            "exp_top_docs": 0,
            "orig_orig": 1,
            "orig_exp_embed": 0.065,
            "exp_embed_exp_embed": 0.010,
            "orig_exp_top_docs": 0,
            "exp_top_docs_exp_top_docs": 0
        }

        self.all_unigrams = {'orig': [('two', [('two', 1)]), ('stand', [('stand', 1)]), ('hands', [('hands', 1)])],
                             'exp_embed': [('two', [('pair', 0.5850129127502441), ('one', 0.576961874961853)]),
                                           ('stand', [('sit', 0.5178298950195312), ('come', 0.3913347125053406)]),
                                           ('hands', [('arm', 0.446733295917511), ('ears', 0.410521537065506)])]}
        self.all_bigrams = {'orig_orig': [(('two', 'stand'), [(('two', 1), ('stand', 1))]),
                                          (('stand', 'hands'), [(('stand', 1), ('hands', 1))])],
                            'orig_exp_embed': [(('two', 'stand'), [(('two', 1), ('sit', 0.5178298950195312)),
                                                                   (('two', 1), ('come', 0.3913347125053406))]), (
                                                   ('stand', 'hands'), [(('stand', 1), ('arm', 0.446733295917511)),
                                                                        (('stand', 1),
                                                                         ('ears', 0.410521537065506))])],
                            'exp_embed_exp_embed': [(('two', 'stand'),
                                                     [(('pair', 0.5850129127502441), ('sit', 0.5178298950195312)),
                                                      (('pair', 0.5850129127502441), ('come', 0.3913347125053406)),
                                                      (('one', 0.576961874961853), ('sit', 0.5178298950195312)),
                                                      (('one', 0.576961874961853), ('come', 0.3913347125053406))]), (
                                                        ('stand', 'hands'),
                                                        [(('sit', 0.5178298950195312), ('arm', 0.446733295917511)),
                                                         (('sit', 0.5178298950195312), ('ears', 0.410521537065506)),
                                                         (('come', 0.3913347125053406), ('arm', 0.446733295917511)),
                                                         (('come', 0.3913347125053406),
                                                          ('ears', 0.410521537065506))])]}

        self.all_fields = {
            'u': '#weight(\n    1#weight(\n      0.90000000000000002220#combine(two)\n      0.59999999999999997780#combine(stand)\n      0.10000000000000000555#combine(hands)\n    )\n    0.065#weight(\n      0.90000000000000002220#combine(pair)\n      0.10000000000000000555#combine(one)\n      0.59999999999999997780#combine(sit)\n      0.29999999999999998890#combine(come)\n      0.10000000000000000555#combine(arm)\n      0.10000000000000000555#combine(ears)\n    )\n  )\n',
            'o': '#weight(\n    1#weight(\n      0.59999999999999997780#od(two stand)\n      0.20000000000000001110#od(stand hands)\n    )\n    0.065#weight(\n      0.29999999999999998890#od(two sit)\n      0.40000000000000002220#od(two come)\n      0.10000000000000000555#od(stand arm)\n      0.10000000000000000555#od(stand ears)\n    )\n    0.01#weight(\n      0.50000000000000000000#od(pair sit)\n      0.50000000000000000000#od(pair come)\n      0.50000000000000000000#od(one sit)\n      0.50000000000000000000#od(one come)\n      0.50000000000000000000#od(sit arm)\n      0.50000000000000000000#od(sit ears)\n      0.50000000000000000000#od(come arm)\n      0.50000000000000000000#od(come ears)\n    )\n  )\n',
            'w': '#weight(\n    1#weight(\n      0.59999999999999997780#uw(two stand)\n      0.20000000000000001110#uw(stand hands)\n    )\n    0.065#weight(\n      0.29999999999999998890#uw(two sit)\n      0.40000000000000002220#uw(two come)\n      0.10000000000000000555#uw(stand arm)\n      0.10000000000000000555#uw(stand ears)\n    )\n    0.01#weight(\n      0.50000000000000000000#uw(pair sit)\n      0.50000000000000000000#uw(pair come)\n      0.50000000000000000000#uw(one sit)\n      0.50000000000000000000#uw(one come)\n      0.50000000000000000000#uw(sit arm)\n      0.50000000000000000000#uw(sit ears)\n      0.50000000000000000000#uw(come arm)\n      0.50000000000000000000#uw(come ears)\n    )\n  )\n'}

    def test_gen_combine_fields_text(self):
        field_weights = {
            "u": 0.8,
            "o": 0.1,
            "w": 0.1
        }

        query_language_modifier = QueryLanguageModifier(self.parameters)
        query_language_modifier.embedding_space.initialize = MagicMock(return_value=None)
        res = query_language_modifier._gen_weighted_fields_text(field_weights, self.all_fields)
        print(res, file=sys.stderr)
        expected_res = """
#weight(
  0.80000#weight(
    1#weight(
      0.90000000000000002220#combine(two)
      0.59999999999999997780#combine(stand)
      0.10000000000000000555#combine(hands)
    )
    0.065#weight(
      0.90000000000000002220#combine(pair)
      0.10000000000000000555#combine(one)
      0.59999999999999997780#combine(sit)
      0.29999999999999998890#combine(come)
      0.10000000000000000555#combine(arm)
      0.10000000000000000555#combine(ears)
    )
  )
  0.10000#weight(
    1#weight(
      0.59999999999999997780#od(two stand)
      0.20000000000000001110#od(stand hands)
    )
    0.065#weight(
      0.29999999999999998890#od(two sit)
      0.40000000000000002220#od(two come)
      0.10000000000000000555#od(stand arm)
      0.10000000000000000555#od(stand ears)
    )
    0.01#weight(
      0.50000000000000000000#od(pair sit)
      0.50000000000000000000#od(pair come)
      0.50000000000000000000#od(one sit)
      0.50000000000000000000#od(one come)
      0.50000000000000000000#od(sit arm)
      0.50000000000000000000#od(sit ears)
      0.50000000000000000000#od(come arm)
      0.50000000000000000000#od(come ears)
    )
  )
  0.10000#weight(
    1#weight(
      0.59999999999999997780#uw(two stand)
      0.20000000000000001110#uw(stand hands)
    )
    0.065#weight(
      0.29999999999999998890#uw(two sit)
      0.40000000000000002220#uw(two come)
      0.10000000000000000555#uw(stand arm)
      0.10000000000000000555#uw(stand ears)
    )
    0.01#weight(
      0.50000000000000000000#uw(pair sit)
      0.50000000000000000000#uw(pair come)
      0.50000000000000000000#uw(one sit)
      0.50000000000000000000#uw(one come)
      0.50000000000000000000#uw(sit arm)
      0.50000000000000000000#uw(sit ears)
      0.50000000000000000000#uw(come arm)
      0.50000000000000000000#uw(come ears)
    )
  )
)
"""
        self.assertEqual(res, expected_res)

    def test_gen_sdm_fields_texts(self):
        query_language_modifier = QueryLanguageModifier(self.parameters)
        query_language_modifier.expanded_sdm.compute_weight_sdm_unigrams = MagicMock(
            side_effect=mock_compute_weight_sdm_grams)
        query_language_modifier.expanded_sdm.compute_weight_sdm_grams = MagicMock(
            side_effect=mock_compute_weight_sdm_grams)
        query_language_modifier._find_all_unigrams = \
            MagicMock(return_value=self.all_unigrams)
        query_language_modifier._find_all_bigrams = \
            MagicMock(return_value=self.all_bigrams)
        query_language_modifier.embedding_space.initialize = MagicMock(return_value=None)
        res = query_language_modifier._gen_sdm_fields_texts("two stand hands")
        expected_res = self.all_fields
        print(res, file=sys.stderr)
        self.assertEqual(res, expected_res)

    def test_keep_cv_queries(self):
        query_language_modifier = QueryLanguageModifier(self.parameters)
        soup = Queries().indri_query_file_2_soup("test_files/indri_query.cfg")
        queries = query_language_modifier._find_all_queries(soup)

        query_numbers = ["INEX_LD-20120121", "QALD2_te-81"]

        query_language_modifier._keep_cv_queries(queries, query_numbers)

        self.assertEqual(str(soup.find_all("query")), """[<query>
<number>INEX_LD-20120121</number>
<text>vietnam food recipes</text>
</query>, <query>
<number>QALD2_te-81</number>
<text>Which books by Kerouac were published by Viking Press</text>
</query>]""")

    def test_get_query_numbers_to_keep(self):
        self.parameters.params["cross_validation"] = {"number_of_folds": 3, "testing_fold": 1}

        query_language_modifier = QueryLanguageModifier(self.parameters)
        soup = Queries().indri_query_file_2_soup("test_files/indri_query.cfg")
        queries = query_language_modifier._find_all_queries(soup)

        res = query_language_modifier._get_query_numbers_to_keep(queries, is_test=True)

        self.assertEqual(res, ['INEX_LD-20120112', 'QALD2_te-81'])

        res = query_language_modifier._get_query_numbers_to_keep(queries, is_test=False)
        self.assertEqual(res, ['INEX_LD-20120111', 'INEX_LD-20120121', 'QALD2_te-82', 'TREC_Entity-20'])

    def test__find_all_bigrams_(self):
        query_language_modifier = QueryLanguageModifier(self.parameters)
        res = query_language_modifier._find_all_bigrams_(self.all_unigrams["orig"], self.all_unigrams["orig"])
        res = list(res)
        print(res, file=sys.stderr)
        self.assertEqual(res, [(('two', 'stand'), [(('two', 1), ('stand', 1))]),
                               (('stand', 'hands'), [(('stand', 1), ('hands', 1))])])

    def test__find_all_bigrams(self):
        query_language_modifier = QueryLanguageModifier(self.parameters)
        res = query_language_modifier._find_all_bigrams(self.all_unigrams)
        print(res, file=sys.stderr)
        self.assertEqual(res, self.all_bigrams)

    def test__find_all_unigrams(self):
        self.parameters.params["word2vec"] = {"upper_threshold": 1, "lower_threshold": 0, "n_max": 2}
        query_language_modifier = QueryLanguageModifier(self.parameters)
        query_language_modifier.embedding_space.initialize()
        res = query_language_modifier._find_all_unigrams("two stand hands")
        print(res, file=sys.stderr)
        self.assertEqual(res, {'orig': [('two', [('two', 1)]), ('stand', [('stand', 1)]), ('hands', [('hands', 1)])],
                               'exp_embed': [('two', [('pair', 0.5850129127502441), ('one', 0.576961874961853),
                                                      ('handful', 0.553887128829956)]), ('stand',
                                                                                         [('sit', 0.5178298950195312),
                                                                                          ('come', 0.3913347125053406),
                                                                                          ('lay', 0.3732541799545288)]),
                                             ('hands', [('arm', 0.446733295917511), ('ears', 0.410521537065506),
                                                        ('hearts', 0.369223415851593)])]})

    def test__gen_sdm_fields_texts(self):
        query_language_modifier = QueryLanguageModifier(self.parameters)
        res = query_language_modifier._find_all_bigrams(self.all_unigrams)
        print(res, file=sys.stderr)
        self.assertEqual(res, {'orig_orig': [(('two', 'stand'), [(('two', 1), ('stand', 1))]),
                                             (('stand', 'hands'), [(('stand', 1), ('hands', 1))])],
                               'orig_exp_embed': [(('two', 'stand'), [(('two', 1), ('sit', 0.5178298950195312)),
                                                                      (('two', 1), ('come', 0.3913347125053406))]), (
                                                      ('stand', 'hands'), [(('stand', 1), ('arm', 0.446733295917511)),
                                                                           (('stand', 1),
                                                                            ('ears', 0.410521537065506))])],
                               'exp_embed_exp_embed': [(('two', 'stand'),
                                                        [(('pair', 0.5850129127502441), ('sit', 0.5178298950195312)),
                                                         (('pair', 0.5850129127502441), ('come', 0.3913347125053406)),
                                                         (('one', 0.576961874961853), ('sit', 0.5178298950195312)),
                                                         (('one', 0.576961874961853), ('come', 0.3913347125053406))]), (
                                                           ('stand', 'hands'),
                                                           [(('sit', 0.5178298950195312), ('arm', 0.446733295917511)),
                                                            (('sit', 0.5178298950195312), ('ears', 0.410521537065506)),
                                                            (('come', 0.3913347125053406), ('arm', 0.446733295917511)),
                                                            (('come', 0.3913347125053406),
                                                             ('ears', 0.410521537065506))])]})


if __name__ == '__main__':
    unittest.main()
