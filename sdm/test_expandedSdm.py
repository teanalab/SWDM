import sys
import unittest

from mock import MagicMock

from parameters.parameters import Parameters
from sdm.expanded_sdm import ExpandedSdm

__author__ = 'Saeid Balaneshin-kordan'
__email__ = "saeid@wayne.edu"
__date__ = 11 / 21 / 16


def mock_compute_weight_sdm_grams(gram, operator):
    del operator
    weight_gram = {
        ('two', 'pair'): 0.9,
        ('two', 'one'): 0.1,
        ('two', 'handful'): 0.05,
        ('stand', 'sit'): 0.6,
        ('stand', 'come'): 0.3,
        ('stand', 'lay'): 0.05,
        ('hands', 'arm'): 0.1,
        ('hands', 'ears'): 0.1,
        ('hands', 'hearts'): 0.02,
        ('two', 'two'): 0.9,
        ('stand', 'stand'): 0.6,
        ('hands', 'hands'): 0.1,
        (('two', 'stand'), ('two', 'stand')): 0.6,
        (('stand', 'hands'), ('stand', 'hands')): 0.2,
        (('two', 'stand'), ('two', 'sit')): 0.3,
        (('two', 'stand'), ('two', 'come')): 0.4,
        (('stand', 'hands'), ('stand', 'arm')): 0.1,
        (('stand', 'hands'), ('stand', 'ears')): 0.1,
        (('two', 'stand'), ('pair', 'sit')): 0.5,
        (('two', 'stand'), ('pair', 'come')): 0.5,
        (('two', 'stand'), ('one', 'sit')): 0.5,
        (('two', 'stand'), ('one', 'come')): 0.5,
        (('stand', 'hands'), ('sit', 'arm')): 0.5,
        (('stand', 'hands'), ('sit', 'ears')): 0.5,
        (('stand', 'hands'), ('come', 'arm')): 0.5,
        (('stand', 'hands'), ('come', 'ears')): 0.5,
    }
    if isinstance(gram[1][0], str):
        gram = (gram[0], gram[1][0])
    else:
        gram = (gram[0], (gram[1][0][0], gram[1][1][0]))
    return weight_gram[gram]


class TestExpandedSdm(unittest.TestCase):
    def setUp(self):
        self.parameters = Parameters()
        self.parameters.params["repo_dir"] = '../index/test_files/index'
        self.parameters.params['feature_parameters'] = {}
        self.parameters.params['feature_parameters']['UnorderedBigramWeights'] = {
            "uw_expression_norm_count": {
                "window_size": 17
            },
            "uw_expression_norm_document_count": {
                "window_size": 17
            },
            "td_uw_expression_norm_count": {
                "window_size": 17,
                "n_top_docs": 10,
            },
            "td_uw_expression_norm_document_count": {
                "window_size": 17,
                "n_top_docs": 10,
            },
            'bigrams_cosine_similarity_with_orig': {}
        }
        self.parameters.params['feature_parameters']['OrderedBigramWeights'] = {
            "od_expression_norm_count": {
                "window_size": 17
            },
            "od_expression_norm_document_count": {
                "window_size": 17
            },
            "td_od_expression_norm_count": {
                "window_size": 17,
                "n_top_docs": 10,
            },
            "td_od_expression_norm_document_count": {
                "window_size": 17,
                "n_top_docs": 10,
            }
        }
        self.parameters.params['features_weights'] = {}
        self.parameters.params['features_weights']['UnorderedBigramWeights'] = {
            "uw_expression_norm_count": 0.33,
            "uw_expression_norm_document_count": 0.33,
            "bigrams_cosine_similarity_with_orig": 0.33,
            'td_uw_expression_norm_document_count': 0.1,
            "td_uw_expression_norm_count": 0.1
        }
        self.parameters.params['features_weights']['OrderedBigramWeights'] = {
            "od_expression_norm_count": 0.33,
            "od_expression_norm_document_count": 0.33,
            "bigrams_cosine_similarity_with_orig": 0.33
        }
        self.parameters.params['feature_parameters']['UnigramWeights'] = {
            "norm_term_count": {
            },
            "norm_document_count": {
            },
            "unigrams_cosine_similarity_with_orig": {
            },
            "td_norm_unigram_count": {
                "n_top_docs": 17
            },
            "td_norm_unigram_document_count": {
                "n_top_docs": 17
            }
        }
        self.parameters.params['features_weights']['UnigramWeights'] = {
            "norm_term_count": 0.33,
            "norm_document_count": 0.33,
            "unigrams_cosine_similarity_with_orig": 0.33,
            "td_norm_unigram_count": 0.05,
            "td_norm_unigram_document_count": 0.1
        }
        self.parameters.params["window_size"] = {}
        self.parameters.params["window_size"]["#od"] = 4
        self.parameters.params["window_size"]["#uw"] = 17
        self.parameters.params["type_weights"] = {"orig": 1,
                                                  "exp_embed": 0.05,
                                                  "exp_top_docs": 0.05}

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

    def test_gen_sdm_grams_field_1_text(self):
        unigrams = self.all_unigrams["exp_embed"]
        expanded_sdm = ExpandedSdm(self.parameters)
        expanded_sdm.compute_weight_sdm_grams = MagicMock(
            side_effect=mock_compute_weight_sdm_grams)
        res = expanded_sdm.gen_sdm_grams_field_1_text(unigrams, "#combine")
        print(res, file=sys.stderr)
        expected_res = """#weight(
      0.90000000000000002220#combine(pair)
      0.10000000000000000555#combine(one)
      0.59999999999999997780#combine(sit)
      0.29999999999999998890#combine(come)
      0.10000000000000000555#combine(arm)
      0.10000000000000000555#combine(ears)
    )
"""
        self.assertEqual(res, expected_res)

        unigrams = self.all_unigrams["orig"]
        expanded_sdm.compute_weight_sdm_unigrams = MagicMock(
            side_effect=mock_compute_weight_sdm_grams)
        res = expanded_sdm.gen_sdm_grams_field_1_text(unigrams, "#combine")
        expected_res = """#weight(
      0.90000000000000002220#combine(two)
      0.59999999999999997780#combine(stand)
      0.10000000000000000555#combine(hands)
    )
"""
        print(res, file=sys.stderr)
        self.assertEqual(res, expected_res)

        print("-" * 50, file=sys.stderr)

        expanded_sdm = ExpandedSdm(self.parameters)
        expanded_sdm.compute_weight_sdm_grams = MagicMock(
            side_effect=mock_compute_weight_sdm_grams)
        res = expanded_sdm.gen_sdm_grams_field_1_text(self.all_bigrams["orig_orig"], "#uw")
        print(res, file=sys.stderr)
        expected_res = """#weight(
      0.59999999999999997780#uw(two stand)
      0.20000000000000001110#uw(stand hands)
    )
"""
        self.assertEqual(res, expected_res)

        res = expanded_sdm.gen_sdm_grams_field_1_text(self.all_bigrams["exp_embed_exp_embed"], "#uw")
        print(res, file=sys.stderr)
        expected_res = """#weight(
      0.50000000000000000000#uw(pair sit)
      0.50000000000000000000#uw(pair come)
      0.50000000000000000000#uw(one sit)
      0.50000000000000000000#uw(one come)
      0.50000000000000000000#uw(sit arm)
      0.50000000000000000000#uw(sit ears)
      0.50000000000000000000#uw(come arm)
      0.50000000000000000000#uw(come ears)
    )
"""
        self.assertEqual(res, expected_res)

    def test_gen_sdm_field_1_text(self):
        expanded_sdm = ExpandedSdm(self.parameters)
        expanded_sdm.compute_weight_sdm_grams = MagicMock(
            side_effect=mock_compute_weight_sdm_grams)
        res = expanded_sdm.gen_sdm_field_1_text(self.all_unigrams, "#combine")
        expected_res = """#weight(
    1#weight(
      0.90000000000000002220#combine(two)
      0.59999999999999997780#combine(stand)
      0.10000000000000000555#combine(hands)
    )
    0.05#weight(
      0.90000000000000002220#combine(pair)
      0.10000000000000000555#combine(one)
      0.59999999999999997780#combine(sit)
      0.29999999999999998890#combine(come)
      0.10000000000000000555#combine(arm)
      0.10000000000000000555#combine(ears)
    )
  )
"""
        print(res, file=sys.stderr)
        self.assertEqual(res, expected_res)

    def test_compute_weight_sdm_grams(self):
        expanded_sdm = ExpandedSdm(self.parameters)
        expanded_sdm.init_top_docs_run_query("a")

        gram_pair = ('two', ('pair', 0.5850129127502441))
        res = expanded_sdm.compute_weight_sdm_grams(gram_pair, "#combine")
        expected_res = 2.783865029925421
        self.assertAlmostEquals(res, expected_res)

        gram_pair = (('two', 'stand'), (('pair', 0.5850129127502441), ('sit', 0.5178298950195312)))
        res = expanded_sdm.compute_weight_sdm_grams(gram_pair, "#uw")
        print(res, file=sys.stderr)
        expected_res = 3.6109552488452574
        self.assertAlmostEquals(res, expected_res)


if __name__ == '__main__':
    unittest.main()
